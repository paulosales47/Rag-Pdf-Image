#!/usr/bin/env bash
# Script de conveniência para rodar o rag-pdf-local no dia a dia.
# Uso: scripts/rag-pdf.sh {start|stop|restart|status|clean|logs}
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
APP_PATH="src/rag_pdf/app/streamlit_app.py"
LOG_FILE="/tmp/rag-pdf-local-streamlit.log"
PID_FILE="/tmp/rag-pdf-local-streamlit.pid"
LM_STUDIO_URL="http://localhost:1234/v1/models"
CRISPEMBED_URL="http://localhost:8081/health"

activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Ambiente virtual não encontrado em $VENV_DIR"
        echo "Rode primeiro: python -m venv .venv && source .venv/bin/activate && make install"
        exit 1
    fi
    # Linux/macOS: .venv/bin/activate — Windows (python -m venv nativo): .venv/Scripts/activate
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/Scripts/activate"
    else
        echo "Não achei bin/activate nem Scripts/activate dentro de $VENV_DIR"
        exit 1
    fi
}

check_lm_studio() {
    if curl -s -o /dev/null -w "%{http_code}" "$LM_STUDIO_URL" 2>/dev/null | grep -q "200"; then
        echo "✓ LM Studio respondendo em $LM_STUDIO_URL"
    else
        echo "⚠ LM Studio não respondeu em $LM_STUDIO_URL"
        echo "  Abra o LM Studio, carregue os modelos e clique em 'Start Server' (aba Developer)."
    fi
}

check_crispembed() {
    if curl -s -o /dev/null -w "%{http_code}" "$CRISPEMBED_URL" 2>/dev/null | grep -q "200"; then
        echo "✓ CrispEmbed respondendo em $CRISPEMBED_URL"
    else
        echo "⚠ CrispEmbed não respondeu em $CRISPEMBED_URL (só necessário pra aba de Imagens)"
        echo "  Suba com: embedding_image_server\\start-server.ps1 (ou crispembed-server --vit siglip-so400m-patch14-384.gguf --port 8081)"
    fi
}

# pgrep/pkill não vêm com o Git for Windows (só com MSYS2 completo), então rastreamos o
# processo por PID salvo em arquivo em vez de casar pela linha de comando.
is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

cmd_start() {
    if is_running; then
        echo "Já está rodando em http://localhost:8501 (use '$0 restart' para reiniciar)."
        exit 0
    fi
    activate_venv
    check_lm_studio
    check_crispembed
    echo "Subindo a interface web..."
    nohup streamlit run "$APP_PATH" --server.headless true > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    disown
    sleep 3
    if is_running; then
        echo "✓ Rodando em http://localhost:8501 (log: $LOG_FILE)"
    else
        echo "✗ Falha ao subir. Confira o log:"
        tail -n 20 "$LOG_FILE" || true
        rm -f "$PID_FILE"
        exit 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "Não estava rodando."
        rm -f "$PID_FILE"
        return
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        is_running || break
        sleep 0.3
    done
    if is_running; then
        kill -9 "$pid" 2>/dev/null || true
        sleep 0.5
    fi
    rm -f "$PID_FILE"
    echo "✓ Aplicação parada."
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_status() {
    if is_running; then
        echo "✓ Interface web rodando em http://localhost:8501"
    else
        echo "✗ Interface web parada"
    fi
    check_lm_studio
    check_crispembed
    activate_venv
    python - <<'EOF'
from rag_pdf.vectorstore.chroma_store import ChromaVectorStore

try:
    store = ChromaVectorStore()
    docs = store.list_documents()
    print(f"Chunks no vector store: {store.count()}")
    print(f"Documentos indexados: {len(docs)}")
    for d in docs:
        prefix = f"{d['folder']}/" if d["folder"] else ""
        print(f"  - {prefix}{d['source']}")
except Exception as exc:  # noqa: BLE001
    print(f"Vector store indisponível: {exc}")
EOF
}

cmd_clean() {
    read -r -p "Isso vai apagar TODOS os documentos indexados (vector store + pastas). Confirma? [y/N] " resp
    case "$resp" in
        [yY]|[yY][eE][sS])
            find data/vectorstore -mindepth 1 -not -name '.gitkeep' -delete
            echo "✓ Base limpa."
            ;;
        *)
            echo "Cancelado."
            ;;
    esac
}

cmd_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "Nenhum log encontrado ainda em $LOG_FILE (a aplicação já rodou pelo menos uma vez?)"
        exit 1
    fi
    tail -f "$LOG_FILE"
}

case "${1:-}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    status) cmd_status ;;
    clean) cmd_clean ;;
    logs) cmd_logs ;;
    *)
        echo "Uso: $0 {start|stop|restart|status|clean|logs}"
        echo ""
        echo "  start    Sobe a interface web (Streamlit) em background"
        echo "  stop     Para a interface web"
        echo "  restart  Reinicia a interface web"
        echo "  status   Mostra se está rodando, conexão com LM Studio e documentos indexados"
        echo "  clean    Apaga todos os documentos indexados do vector store (pede confirmação)"
        echo "  logs     Acompanha o log da interface web em tempo real"
        exit 1
        ;;
esac
