#!/bin/bash
# Script to upload to TestPyPI with token as parameter

echo "=== LLMShield TestPyPI Upload Script (with token) ==="
echo

# Auto-detect version from pyproject.toml
VERSION=$(grep -E '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

if [ -z "$VERSION" ]; then
    echo "❌ Error: Could not detect version from pyproject.toml"
    exit 1
fi

echo "🔍 Detected version: $VERSION"
echo

# Check if token is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your TestPyPI token as an argument"
    echo "Usage: $0 <your-testpypi-token>"
    echo
    echo "Get your token from: https://test.pypi.org/manage/account/#api-tokens"
    exit 1
fi

# Check if dist files exist
if [ ! -f "dist/llmshield-${VERSION}-py3-none-any.whl" ] || [ ! -f "dist/llmshield-${VERSION}.tar.gz" ]; then
    echo "❌ Error: Distribution files not found. Run 'make build' first."
    exit 1
fi

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo "❌ Error: twine is not installed. Install it with: pip install twine"
    exit 1
fi

echo "📦 Found distribution files:"
ls -la dist/llmshield-${VERSION}*
echo

echo "📤 Uploading to TestPyPI..."
echo

# Upload to TestPyPI using token
TWINE_USERNAME=__token__ TWINE_PASSWORD=$1 twine upload --repository-url https://test.pypi.org/legacy/ dist/llmshield-${VERSION}*

if [ $? -eq 0 ]; then
    echo
    echo "✅ Upload successful!"
    echo
    echo "To test installation in another project, run:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmshield==${VERSION}"
    echo
    echo "View your package at: https://test.pypi.org/project/llmshield/${VERSION}/"
else
    echo
    echo "❌ Upload failed. Please check your token and try again."
fi
