<#
.SYNOPSIS
    Sobe o crispembed-server local (build CUDA) servindo o modelo SigLIP, na porta
    configurada em CRISPEMBED_BASE_URL (.env).

.DESCRIPTION
    Roda em primeiro plano — feche a janela ou Ctrl+C pra parar. Requer ter rodado
    setup.ps1 antes (gera o executável) e ter o .gguf em models\.
#>

param(
    [int]$Port = 8081,
    [string]$ModelPath = (Join-Path $PSScriptRoot "..\models\siglip-so400m-patch14-384.gguf")
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$cloneDir = Join-Path $root "CrispEmbed"

if (-not (Test-Path $cloneDir)) {
    Write-Error "CrispEmbed ainda não foi buildado. Rode primeiro: .\embedding_image_server\setup.ps1"
    exit 1
}

$exe = Get-ChildItem -Path $cloneDir -Recurse -Filter "crispembed-server*.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $exe) {
    Write-Error (
        "Não encontrei 'crispembed-server*.exe' dentro de $cloneDir. Rode " +
        ".\embedding_image_server\setup.ps1 e confira se o build terminou sem erros."
    )
    exit 1
}

# O executável é linkado dinamicamente contra a runtime CUDA (ex.: cublas64_13.dll), que
# vive na pasta bin do CUDA Toolkit. O instalador do CUDA adiciona essa pasta ao PATH da
# máquina (registro), mas esta sessão de PowerShell pode ter sido aberta antes disso — sem
# isso no PATH, o processo falha instantaneamente com STATUS_DLL_NOT_FOUND (0xC0000135),
# sem imprimir nenhuma mensagem de erro.
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
$cudaVersionDir = Get-ChildItem $cudaRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($cudaVersionDir) {
    # Em versões recentes do CUDA Toolkit (ex.: 13.3), as DLLs de runtime (cublas64_*.dll,
    # cudart64_*.dll etc.) ficam em bin\x64, não direto em bin — o instalador coloca as duas
    # pastas no PATH da máquina, então replicamos aqui pro caso desta sessão estar com o PATH
    # desatualizado.
    $cudaBin = Join-Path $cudaVersionDir.FullName "bin"
    $cudaBinX64 = Join-Path $cudaBin "x64"
    $cudaDirs = @($cudaBin)
    if (Test-Path $cudaBinX64) { $cudaDirs += $cudaBinX64 }
    $env:Path = "$($cudaDirs -join ';');$env:Path"
}

$resolvedModelPath = [System.IO.Path]::GetFullPath($ModelPath)
if (-not (Test-Path $resolvedModelPath)) {
    Write-Error (
        "Modelo não encontrado em $resolvedModelPath. Baixe " +
        "siglip-so400m-patch14-384.gguf (repo cstr/siglip-so400m-patch14-384-GGUF no Hugging " +
        "Face) e salve em models\siglip-so400m-patch14-384.gguf."
    )
    exit 1
}

Write-Host "== Subindo crispembed-server ==" -ForegroundColor Cyan
Write-Host "Executável: $($exe.FullName)"
Write-Host "Modelo:     $resolvedModelPath"
Write-Host "Porta:      $Port"
Write-Host ""

# SigLIP/CLIP é um modelo "ViT standalone" no CrispEmbed — precisa entrar via --vit
# (endpoint POST /vit/encode), não via -m (que carrega um modelo de texto/BERT e falha
# com "missing token_embd.weight" para um encoder de imagem puro).
& $exe.FullName --vit $resolvedModelPath --port $Port
