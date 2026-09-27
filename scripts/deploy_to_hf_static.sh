#!/bin/bash
# scripts/deploy_to_hf_static.sh
# Deploys FloraGuard directly to a free Hugging Face Static Space

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_USERNAME=${1:-"TemurbekHamzaev"}
SPACE_NAME=${2:-"FloraGuardAI"}
HF_TOKEN=$3

if [ -n "$HF_TOKEN" ]; then
    REMOTE_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
    DISPLAY_URL="https://${HF_USERNAME}:***@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
else
    REMOTE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
    DISPLAY_URL="${REMOTE_URL}"
fi

echo "=================================================================="
echo "🌿 FLORAGUARD AI — DEPLOYING TO HUGGING FACE STATIC SPACE"
echo "• Target Space:  ${DISPLAY_URL}"
echo "• Hosting Mode:  100% Free Static CDN (Zero Compute Costs / No RAM Limit)"
echo "=================================================================="

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "📦 Packaging pure web files (HTML, CSS, JS, Disease Knowledge Base)..."
cp "$PROJECT_ROOT/app/frontend/index.html" "$TMP_DIR/"
cp "$PROJECT_ROOT/app/frontend/style.css" "$TMP_DIR/"
cp "$PROJECT_ROOT/app/frontend/app.js" "$TMP_DIR/"
cp "$PROJECT_ROOT/app/frontend/disease_info.js" "$TMP_DIR/"

cd "$TMP_DIR"
git init -b main
git config user.name "${HF_USERNAME}"
git config user.email "deploy@floraguard.ai"
git add .
git commit -m "Deploy FloraGuard AI Static Web App"

echo "🚀 Pushing directly to Hugging Face Space repository..."
if [ -z "$HF_TOKEN" ]; then
    echo "💡 Enter Username: ${HF_USERNAME}"
    echo "💡 Enter Password: (Paste your Hugging Face Token starting with hf_...)"
fi
git push "${REMOTE_URL}" main --force

echo ""
echo "=================================================================="
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "Your live public web app is available at:"
echo "👉 https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo "=================================================================="
