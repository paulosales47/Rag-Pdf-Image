#!/usr/bin/env bash
# Script de conveniência para rodar o rag-pdf-local no dia a dia.
# Uso: scripts/rag-pdf.sh {start|stop|restart|status|clean|logs}
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
APP_PATH="src/rag_pdf/app/streamlit_app.py"
LOG_FILE="/tmp/rag-pdf-local-streamlit.log"
LM_STUDIO_URL="http://localhost:1234/v1/models"

activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Ambiente virtual não encontrado em $VENV_DIR"
        echo "Rode primeiro: python -m venv .venv && source .venv/bin/activate && make install"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
}

check_lm_studio() {
    if curl -s -o /dev/null -w "%{http_code}" "$LM_STUDIO_URL" 2>/dev/null | grep -q "200"; then
        echo "✓ LM Studio respondendo em $LM_STUDIO_URL"
    else
        echo "⚠ LM Studio não respondeu em $LM_STUDIO_URL"
        echo "  Abra o LM Studio, carregue os modelos e clique em 'Start Server' (aba Developer)."
    fi
}

is_running() {
    pgrep -f "streamlit run $APP_PATH" > /dev/null 2>&1
}

cmd_start() {
    if is_running; then
        echo "Já está rodando em http://localhost:8501 (use '$0 restart' para reiniciar)."
        exit 0
    fi
    activate_venv
    check_lm_studio
    echo "Subindo a interface web..."
    nohup streamlit run "$APP_PATH" --server.headless true > "$LOG_FILE" 2>&1 &
    disown
    sleep 3
    if is_running; then
        echo "✓ Rodando em http://localhost:8501 (log: $LOG_FILE)"
    else
        echo "✗ Falha ao subir. Confira o log:"
        tail -n 20 "$LOG_FILE" || true
        exit 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "Não estava rodando."
        return
    fi
    pkill -f "streamlit run $APP_PATH" 2>/dev/null || true
    for _ in $(seq 1 10); do
        is_running || break
        sleep 0.3
    done
    if is_running; then
        pkill -9 -f "streamlit run $APP_PATH" 2>/dev/null || true
        sleep 0.5
    fi
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
    activate_venv
    python3 - <<'EOF'
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
