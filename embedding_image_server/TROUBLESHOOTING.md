# Diário de bugs: build e execução do CrispEmbed (SigLIP/CUDA)

Registro dos problemas reais encontrados ao buildar e rodar o CrispEmbed com CUDA nesta
máquina (Windows, VS Build Tools 2026, CUDA Toolkit 13.3, RTX 4080), na ordem em que
apareceram. Todos já estão corrigidos em `setup.ps1` / `start-server.ps1` / `image_embedder.py`
— este arquivo existe pra explicar *por quê* essas correções estão lá, caso alguém precise
reproduzir o processo numa máquina diferente e tropece em algo parecido.

## 1. `cl`/`cmake`/`nvcc` "não instalados" (mas estavam)

**Sintoma:** rodar `cl`, `cmake`, `nvcc` num PowerShell aberto normalmente (não um "Developer
PowerShell") não encontra nenhum dos três, mesmo com Visual Studio Build Tools e CUDA Toolkit
recém-instalados.

**Causa:**
- `cl.exe` e o CMake que acompanha o Visual Studio **nunca** entram no `PATH` global — só ficam
  disponíveis dentro de um "Developer Command Prompt/PowerShell", que roda um script
  (`vcvarsall.bat` / `Launch-VsDevShell.ps1`) pra montar esse ambiente na hora.
- `nvcc` (CUDA Toolkit) o instalador **adiciona ao `PATH` da máquina** (registro do Windows),
  mas uma sessão de terminal já aberta **antes** da instalação não vê essa mudança até ser
  reaberta — o processo herdou o `PATH` antigo na hora em que nasceu.

**Solução:** `setup.ps1` não exige rodar de um Developer Prompt. Em vez disso:
1. Localiza a instalação do Visual Studio via `vswhere.exe` (caminho fixo:
   `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe`).
2. Importa o ambiente de build **na sessão atual** chamando
   `<VS>\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64` — isso ajusta
   `$env:Path`/`INCLUDE`/`LIB` do processo PowerShell já em execução (não abre um processo novo).
   Comprovado: esse script também bota o CMake embutido do VS no `PATH` de brinde.
3. Resolve a pasta de instalação do CUDA mais recente em
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\` e prepende ao `$env:Path` da sessão
   manualmente, sem depender do valor herdado.

## 2. `build-cuda.bat` diz "Visual Studio C++ build tools not found" com tudo instalado

**Sintoma:** com `cl`/`cmake`/`nvcc` já funcionando (pós item 1), o script oficial de build do
próprio CrispEmbed (`build-cuda.bat`) ainda falhava logo no início:
```
[ERROR] Visual Studio C++ build tools not found.
```

**Causa:** bug no próprio `build-cuda.bat` — ele chama:
```bat
"!vswhere!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```
Sem a flag `-products *`, o `vswhere` **só considera as SKUs Community/Professional/Enterprise**
por padrão — a SKU "Build Tools" (que é o que está instalado nesta máquina) fica de fora da
busca. Resultado: `vs_path` fica vazio mesmo com o Build Tools instalado e com o componente
`VC.Tools.x86.x64` presente.

Esse é um bug clássico e bem documentado do `vswhere` (a omissão de `-products *` é o erro mais
comum em scripts de terceiros que usam essa ferramenta) — não é nada específico desta máquina.

**Solução:** `setup.ps1`, depois de clonar/atualizar o repo, faz um patch de uma linha só na
cópia clonada localmente (**não** é upstream, não é versionado por este repo — é regerado a
cada clone):
```powershell
$patched = $original -replace '"!vswhere!" -latest -requires', '"!vswhere!" -latest -products * -requires'
```
Idempotente — só reescreve o arquivo se o padrão antigo ainda estiver lá.

## 3. Executável builda com sucesso, mas morre instantaneamente sem nenhuma mensagem

**Sintoma:** `build-cuda.bat` termina com `[SUCCESS] CUDA build complete!`, o
`crispembed-server.exe` existe no disco, mas rodá-lo (de qualquer jeito — direto, ou via
`start-server.ps1`) encerra **imediatamente**, sem imprimir nada, com código de saída
`-1073741515` (`0xC0000135` = `STATUS_DLL_NOT_FOUND`).

**Causa:** o `crispembed-server.exe` foi linkado **dinamicamente** contra a runtime CUDA
(confirmado com `dumpbin /dependents`: depende de `cublas64_13.dll`). No **CUDA Toolkit 13.x**,
essas DLLs (`cublas64_13.dll`, `cudart64_13.dll`, etc.) ficam em `bin\x64\`, **não** direto em
`bin\` como em versões mais antigas do Toolkit — e o `Windows` falha ao carregar o `.exe` sem
imprimir erro nenhum quando uma dependência dinâmica não é encontrada (crash silencioso, não é
um erro do CrispEmbed).

**Como descobri:** rodando `dumpbin /dependents crispembed-server.exe` (ferramenta que já vem
com o MSVC instalado) pra listar as DLLs que o executável precisa, e comparando com o conteúdo
real de `CUDA\v13.3\bin\` (vazio de DLLs de runtime) vs `CUDA\v13.3\bin\x64\` (lá estavam).

**Solução:** os dois scripts (`setup.ps1` e `start-server.ps1`) agora resolvem a pasta CUDA mais
recente e prependem **tanto** `bin\` quanto `bin\x64\` (quando existe) ao `$env:Path` da sessão
antes de compilar/rodar qualquer coisa.

## 4. Modelo carrega, mas falha com "missing token_embd.weight"

**Sintoma:** com o `.exe` finalmente rodando, subir o servidor com
`crispembed-server.exe -m siglip-so400m-patch14-384.gguf --port 8081` (a forma "óbvia", e a que
a documentação pública do projeto sugere) resultava em:
```
crispembed: missing token_embd.weight
Failed to load model '...\siglip-so400m-patch14-384.gguf'
```
mesmo com o arquivo `.gguf` correto, íntegro, e com metadados válidos (confirmei lendo os
metadados do GGUF diretamente com a lib Python `gguf` — `general.architecture = "vit"`,
`vit.hidden_size = 1152`, `vit.num_hidden_layers = 27`, etc., tudo batendo com o SigLIP
so400m/14 esperado).

**Causa:** o CrispEmbed trata modelos "ViT standalone" (SigLIP/CLIP, encoders de imagem puros,
sem torre de texto) como uma **categoria de modelo totalmente separada** dos modelos de
texto/embedding (BERT-like). A flag `-m` carrega exclusivamente o modelo de texto "primário" —
que exige um tensor `token_embd.weight` (tabela de embedding de tokens) que um encoder de
imagem simplesmente não tem, daí o erro. O caminho de carregamento correto pra um ViT standalone
é uma função C interna completamente diferente (`crispembed_vit_init`), acionada apenas por uma
flag **separada**: `--vit <arquivo.gguf>`.

**Como descobri:** lendo o código-fonte do servidor (`examples/server/server.cpp`) direto do
clone local, já que a documentação pública (README/HF model card) não menciona essa distinção e
sugere `-m` para tudo. Achei a flag `--vit`, a inicialização `crispembed_vit_init`, e o endpoint
que ela habilita.

**Solução:** `start-server.ps1` sobe o servidor com `--vit <modelo>` em vez de `-m <modelo>`.

## 5. Endpoint e formato de resposta errados no cliente Python

**Sintoma:** antes de ter um servidor funcionando de verdade pra testar, `image_embedder.py`
foi escrito com base em documentação de terceiros (HF model card do GGUF, README do projeto)
que citava `POST /image` com `{"image": "..."}` como forma de gerar embedding de imagem.

**Causa:** essa documentação estava desatualizada/imprecisa. O endpoint real, específico pra
ViT standalone, é outro.

**Como descobri:** mesma leitura de `examples/server/server.cpp` do item 4 — o handler real é:
```cpp
svr.Post("/vit/encode", [&](...) {
    // request:  {"image": "/path/to/image.jpg"}
    // response: {"embedding": [...], "dim": N}
});
```

**Solução:** `image_embedder.py` chama `POST /vit/encode` (não `/image`) e lê a chave
`"embedding"` da resposta — confirmado batendo certo com um teste real (`embed_image()` contra
o servidor rodando, devolvendo um vetor de 1152 floats).

## Resumo — o que aprender daqui pra próxima vez

1. **Ferramentas MSVC "instaladas mas não encontradas"** quase sempre é sessão de terminal
   aberta antes da instalação, ou VS Build Tools que nunca entra no `PATH` global por design —
   solução é importar o ambiente (`Launch-VsDevShell.ps1`) na sessão atual, não reabrir o
   terminal e torcer.
2. **`vswhere` sem `-products *`** é o bug mais comum em scripts de terceiros que dependem dele
   — sempre suspeitar disso primeiro quando "Visual Studio not found" aparece com o VS
   claramente instalado.
3. **Crash silencioso e instantâneo de um `.exe` recém-buildado** (sem nenhuma mensagem, código
   de saída bizarro tipo `-1073741515`) é quase sempre DLL dinâmica faltando no `PATH` —
   `dumpbin /dependents` (vem com o MSVC) resolve em segundos.
4. **Não confiar cegamente em documentação de projetos de terceiros em desenvolvimento ativo**
   (READMEs, model cards) para o formato exato de CLI/API — quando algo essencial falha de um
   jeito que a doc não prevê, ler o código-fonte real (que já está clonado localmente) é mais
   rápido e mais confiável do que tentar adivinhar variações do comando.
