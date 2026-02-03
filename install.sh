#!/bin/bash
#
# warp-autocontinue Installation Script
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_SCRIPT="$SCRIPT_DIR/warp_autocontinue.py"

echo "Installing warp-autocontinue..."

chmod +x "$CLI_SCRIPT"

if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
  ln -sf "$CLI_SCRIPT" /usr/local/bin/warp-autocontinue
  echo "✓ Installed warp-autocontinue to /usr/local/bin/warp-autocontinue"
else
  mkdir -p "$HOME/.local/bin"
  ln -sf "$CLI_SCRIPT" "$HOME/.local/bin/warp-autocontinue"
  echo "✓ Installed warp-autocontinue to ~/.local/bin/warp-autocontinue"

  if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "⚠️  ~/.local/bin is not in your PATH."
    echo "   Add this to your ~/.zshrc or ~/.bashrc:"
    echo ""
    echo '   export PATH="$HOME/.local/bin:$PATH"'
    echo ""
  fi
fi

if command -v warp-autocontinue &> /dev/null; then
  echo ""
  echo "✓ Installation complete!"
  echo ""
  echo "Try it out:"
  echo "  warp-autocontinue once"
  echo "  warp-autocontinue run"
else
  echo ""
  echo "Installation complete, but 'warp-autocontinue' command not found in PATH."
  echo "You may need to restart your terminal or add the install directory to PATH."
fi
