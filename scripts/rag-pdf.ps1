<#
.SYNOPSIS
    Script de conveniência para rodar o rag-pdf-local no dia a dia (Windows/PowerShell).

.DESCRIPTION
    Equivalente nativo do scripts/rag-pdf.sh (mantido para Linux/macOS). Existe porque rodar o
    .sh a partir do PowerShell no Windows tem várias pegadinhas: a associação de arquivo `.sh`
    abre uma janela separada do Git Bash sem mostrar nada no terminal atual; e mesmo chamando
    via `bash scripts\rag-pdf.sh`, o bash interpreta `\` como escape (não como separador de
    caminho), então `.\scripts\rag-pdf.sh` vira um caminho quebrado. Este script roda 100% em
    PowerShell nativo, sem depender do bash.

.PARAMETER Command
    start | stop | restart | status | clean | logs

.EXAMPLE
    .\scripts\rag-pdf.ps1 start
#>

param(
    [Parameter(Position = 0)]
    [string]$Command = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$AppPath = "src\rag_pdf\app\streamlit_app.py"
$LogFile = Join-Path $env:TEMP "rag-pdf-local-streamlit.log"
$ErrLogFile = Join-Path $env:TEMP "rag-pdf-local-streamlit.err.log"
$PidFile = Join-Path $env:TEMP "rag-pdf-local-streamlit.pid"
$LmStudioUrl = "http://localhost:1234/v1/models"
$CrispembedUrl = "http://localhost:8081/health"
$AppUrl = "http://localhost:8501"

function Test-VenvExists {
    if (-not (Test-Path $PythonExe)) {
        Write-Host "Ambiente virtual não encontrado em $VenvDir"
        Write-Host 'Rode primeiro: python -m venv .venv ; .venv\Scripts\Activate.ps1 ; pip install -e ".[dev,app]"'
        exit 1
    }
}

function Test-LmStudio {
    try {
        $resp = Invoke-WebRequest -Uri $LmStudioUrl -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "✓ LM Studio respondendo em $LmStudioUrl" -ForegroundColor Green
            return
        }
    } catch {}
    Write-Host "⚠ LM Studio não respondeu em $LmStudioUrl" -ForegroundColor Yellow
    Write-Host "  Abra o LM Studio, carregue os modelos e clique em 'Start Server' (aba Developer)."
}

function Test-Crispembed {
    try {
        $resp = Invoke-WebRequest -Uri $CrispembedUrl -TimeoutSec 3 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            Write-Host "✓ CrispEmbed respondendo em $CrispembedUrl" -ForegroundColor Green
            return
        }
    } catch {}
    Write-Host "⚠ CrispEmbed não respondeu em $CrispembedUrl (só necessário pra aba de Imagens)" -ForegroundColor Yellow
    Write-Host "  Suba com: embedding_image_server\start-server.ps1"
}

function Get-RunningProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $storedId = Get-Content $PidFile -ErrorAction SilentlyContinue
    if (-not $storedId) { return $null }
    return Get-Process -Id $storedId -ErrorAction SilentlyContinue
}

function Cmd-Start {
    if (Get-RunningProcess) {
        Write-Host "Já está rodando em $AppUrl (use '.\scripts\rag-pdf.ps1 restart' para reiniciar)."
        return
    }
    Test-VenvExists
    Test-LmStudio
    Test-Crispembed
    Write-Host "Subindo a interface web..."
    $proc = Start-Process -FilePath $PythonExe `
        -ArgumentList @("-m", "streamlit", "run", $AppPath, "--server.headless", "true") `
        -RedirectStandardOutput $LogFile -RedirectStandardError $ErrLogFile `
        -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $proc.Id
    Start-Sleep -Seconds 3
    if (Get-RunningProcess) {
        Write-Host "✓ Rodando em $AppUrl (log: $LogFile)" -ForegroundColor Green
    } else {
        Write-Host "✗ Falha ao subir. Confira o log:" -ForegroundColor Red
        Get-Content $ErrLogFile -Tail 20 -ErrorAction SilentlyContinue
        Get-Content $LogFile -Tail 20 -ErrorAction SilentlyContinue
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        exit 1
    }
}

function Cmd-Stop {
    $proc = Get-RunningProcess
    if (-not $proc) {
        Write-Host "Não estava rodando."
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        return
    }
    Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    Write-Host "✓ Aplicação parada." -ForegroundColor Green
}

function Cmd-Restart {
    Cmd-Stop
    Start-Sleep -Seconds 1
    Cmd-Start
}

function Cmd-Status {
    if (Get-RunningProcess) {
        Write-Host "✓ Interface web rodando em $AppUrl" -ForegroundColor Green
    } else {
        Write-Host "✗ Interface web parada" -ForegroundColor Red
    }
    Test-LmStudio
    Test-Crispembed
    Test-VenvExists
    $statusScript = @'
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore

try:
    store = ChromaVectorStore()
    docs = store.list_documents()
    print(f"Chunks no vector store: {store.count()}")
    print(f"Documentos indexados: {len(docs)}")
    for d in docs:
        prefix = f"{d['folder']}/" if d["folder"] else ""
        print(f"  - {prefix}{d['source']}")
except Exception as exc:
    print(f"Vector store indisponível: {exc}")
'@
    $statusScript | & $PythonExe -
}

function Cmd-Clean {
    $resp = Read-Host "Isso vai apagar TODOS os documentos indexados (vector store + pastas). Confirma? [y/N]"
    if ($resp -match '^[yY]') {
        Get-ChildItem "data\vectorstore" -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne ".gitkeep" } |
            Remove-Item -Recurse -Force
        Write-Host "✓ Base limpa." -ForegroundColor Green
    } else {
        Write-Host "Cancelado."
    }
}

function Cmd-Logs {
    if (-not (Test-Path $LogFile)) {
        Write-Host "Nenhum log encontrado ainda em $LogFile (a aplicação já rodou pelo menos uma vez?)"
        exit 1
    }
    Get-Content $LogFile -Tail 20 -Wait
}

function Show-Usage {
    Write-Host "Uso: .\scripts\rag-pdf.ps1 {start|stop|restart|status|clean|logs}"
    Write-Host ""
    Write-Host "  start    Sobe a interface web (Streamlit) em background"
    Write-Host "  stop     Para a interface web"
    Write-Host "  restart  Reinicia a interface web"
    Write-Host "  status   Mostra se está rodando, conexão com LM Studio/CrispEmbed e documentos indexados"
    Write-Host "  clean    Apaga todos os documentos indexados do vector store (pede confirmação)"
    Write-Host "  logs     Acompanha o log da interface web em tempo real"
}

switch ($Command) {
    "start"   { Cmd-Start }
    "stop"    { Cmd-Stop }
    "restart" { Cmd-Restart }
    "status"  { Cmd-Status }
    "clean"   { Cmd-Clean }
    "logs"    { Cmd-Logs }
    default   { Show-Usage; exit 1 }
}
