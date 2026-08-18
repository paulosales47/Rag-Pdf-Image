<#
.SYNOPSIS
    Clona e builda o CrispEmbed (github.com/CrispStrobe/CrispEmbed) com suporte a CUDA,
    para servir o embedding visual SigLIP usado pela aba de Imagens.

.DESCRIPTION
    Roda uma vez (ou de novo pra atualizar/rebuildar). Deixa o resultado em
    embedding_image_server\CrispEmbed\ (fora do controle de versão deste repo).
    Veja README.md nesta pasta para os pré-requisitos.
#>

param(
    [string]$RepoUrl = "https://github.com/CrispStrobe/CrispEmbed.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cloneDir = Join-Path $root "CrispEmbed"
$buildDir = Join-Path $cloneDir "build"

function Test-CommandExists($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "== Verificando pré-requisitos ==" -ForegroundColor Cyan

if (-not (Test-CommandExists "git")) {
    Write-Error "git não encontrado no PATH. Instale o Git antes de continuar."
    exit 1
}

# O instalador do Visual Studio não coloca o próprio vswhere.exe no PATH — várias
# ferramentas (inclusive o Launch-VsDevShell.ps1 usado abaixo) esperam encontrá-lo lá.
$vswhereDir = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
if ((Test-Path (Join-Path $vswhereDir "vswhere.exe")) -and -not (Test-CommandExists "vswhere.exe")) {
    $env:Path = "$vswhereDir;$env:Path"
}

# cmake e cl.exe vêm com o Visual Studio Build Tools, mas não são adicionados ao PATH
# global pelo instalador (só ficam disponíveis dentro de um "Developer Command Prompt").
# Em vez de exigir que o script seja rodado de lá, localizamos a instalação via vswhere
# e importamos o ambiente de build (Launch-VsDevShell.ps1) diretamente nesta sessão.
if (-not (Test-CommandExists "cl")) {
    Write-Host "cl.exe não está no PATH desta sessão — localizando o Visual Studio via vswhere..."
    $vswhere = Join-Path $vswhereDir "vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error (
            "vswhere.exe não encontrado — o Visual Studio Build Tools (workload 'Desktop " +
            "development with C++') não parece estar instalado. Veja README.md nesta pasta."
        )
        exit 1
    }

    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) {
        Write-Error (
            "Nenhuma instalação do Visual Studio com o workload de C++ (VC.Tools.x86.x64) foi " +
            "encontrada. Confira se 'Desktop development with C++' está marcado no VS Installer."
        )
        exit 1
    }

    $devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
    Write-Host "Importando ambiente de build de: $vsPath"
    & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

    if (-not (Test-CommandExists "cl")) {
        Write-Error "cl.exe continua fora do PATH mesmo após Launch-VsDevShell.ps1. Abortando."
        exit 1
    }

    # O CMake que acompanha o Build Tools também não entra no PATH pelo dev shell; adiciona à mão.
    $bundledCmakeDir = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if ((Test-Path $bundledCmakeDir) -and -not (Test-CommandExists "cmake")) {
        Write-Host "Adicionando o CMake do Visual Studio ao PATH desta sessão: $bundledCmakeDir"
        $env:Path = "$bundledCmakeDir;$env:Path"
    }
}

if (-not (Test-CommandExists "cmake")) {
    Write-Error "cmake não encontrado (nem no PATH, nem junto do Visual Studio). Veja README.md."
    exit 1
}

# O instalador do CUDA Toolkit atualiza o PATH da máquina (registro), mas esta sessão do
# PowerShell pode ter sido aberta antes disso e não ver a mudança até reiniciar — resolve
# a instalação mais recente diretamente e injeta no PATH se `nvcc` ainda não for encontrado.
if (-not (Test-CommandExists "nvcc")) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $cudaVersionDir = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $cudaVersionDir) {
        Write-Error (
            "nvcc não encontrado e nenhuma instalação em '$cudaRoot'. Instale o CUDA Toolkit " +
            "(veja README.md nesta pasta)."
        )
        exit 1
    }
    # Em versões recentes do CUDA Toolkit (ex.: 13.3), as DLLs de runtime (cublas64_*.dll,
    # cudart64_*.dll etc.) ficam em bin\x64, não direto em bin — inclui as duas.
    $cudaBin = Join-Path $cudaVersionDir.FullName "bin"
    $cudaBinX64 = Join-Path $cudaBin "x64"
    $cudaDirs = @($cudaBin)
    if (Test-Path $cudaBinX64) { $cudaDirs += $cudaBinX64 }
    Write-Host "Adicionando CUDA Toolkit ao PATH desta sessão: $($cudaDirs -join ', ')"
    $env:Path = "$($cudaDirs -join ';');$env:Path"
    $env:CUDACXX = Join-Path $cudaBin "nvcc.exe"

    if (-not (Test-CommandExists "nvcc")) {
        Write-Error "nvcc ainda não encontrado após ajustar o PATH. Confira a instalação do CUDA Toolkit."
        exit 1
    }
}

Write-Host "✓ git, cl.exe, cmake e nvcc disponíveis nesta sessão." -ForegroundColor Green

Write-Host "== Clonando/atualizando CrispEmbed em $cloneDir ==" -ForegroundColor Cyan
if (Test-Path $cloneDir) {
    Push-Location $cloneDir
    git fetch origin
    git checkout $Branch
    git pull origin $Branch
    Pop-Location
} else {
    git clone --branch $Branch --depth 1 $RepoUrl $cloneDir
}

Write-Host "== Buildando com CUDA ==" -ForegroundColor Cyan
Push-Location $cloneDir
try {
    $buildCudaScript = Join-Path $cloneDir "build-cuda.bat"
    if (Test-Path $buildCudaScript) {
        # Bug conhecido do vswhere: sem "-products *" ele só considera Community/Professional/
        # Enterprise, ignorando a SKU "Build Tools" — nessa máquina isso faz o build-cuda.bat
        # original relatar "Visual Studio C++ build tools not found" mesmo com tudo instalado.
        # Corrige a cópia clonada localmente (idempotente; não é versionada por este repo).
        $original = Get-Content $buildCudaScript -Raw
        $patched = $original -replace '"!vswhere!" -latest -requires', '"!vswhere!" -latest -products * -requires'
        if ($patched -ne $original) {
            Write-Host "Aplicando correção conhecida (vswhere -products *) em build-cuda.bat local..."
            Set-Content -Path $buildCudaScript -Value $patched -NoNewline
        }

        Write-Host "Usando o script de build oficial: build-cuda.bat"
        cmd /c $buildCudaScript
        if ($LASTEXITCODE -ne 0) {
            throw "build-cuda.bat terminou com código $LASTEXITCODE"
        }
    } else {
        Write-Host "build-cuda.bat não encontrado nesta versão do repo; buildando via CMake direto."
        cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build build --config Release -j
    }
} finally {
    Pop-Location
}

Write-Host "== Procurando o executável gerado ==" -ForegroundColor Cyan
$exe = Get-ChildItem -Path $cloneDir -Recurse -Filter "crispembed-server*.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($exe) {
    Write-Host "✓ Build concluído: $($exe.FullName)" -ForegroundColor Green
    Write-Host "Rode agora: .\embedding_image_server\start-server.ps1"
} else {
    Write-Warning (
        "Build rodou, mas não encontrei 'crispembed-server*.exe' dentro de $cloneDir. " +
        "Confira a saída acima em busca de erros, ou procure manualmente o executável gerado " +
        "e ajuste o caminho em start-server.ps1."
    )
}
