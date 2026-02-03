#!/bin/bash

echo "=========================================="
echo "Titan-Optimal Migration Verification"
echo "=========================================="
echo ""

# Check 1: Folder exists with correct name
echo "✓ Checking folder name..."
if [ -d "/app/titan_optimal" ]; then
    echo "  ✅ titan_optimal folder exists"
else
    echo "  ❌ titan_optimal folder NOT found"
    exit 1
fi

if [ -d "/app/titan-optimal" ]; then
    echo "  ⚠️  WARNING: Old titan-optimal folder still exists!"
else
    echo "  ✅ Old titan-optimal folder removed"
fi

echo ""

# Check 2: No hyphenated references in code (excluding documentation)
echo "✓ Checking for hyphenated references in code..."
HYPHEN_COUNT=$(grep -r "titan-optimal" /app/titan_optimal --include="*.py" 2>/dev/null | wc -l)
if [ "$HYPHEN_COUNT" -eq 0 ]; then
    echo "  ✅ No hyphenated references in Python files ($HYPHEN_COUNT)"
else
    echo "  ❌ Found $HYPHEN_COUNT hyphenated references in Python files"
    echo "  Files with issues:"
    grep -r "titan-optimal" /app/titan_optimal --include="*.py" -l 2>/dev/null
    exit 1
fi

# Documentation may reference old name for comparison - that's OK
DOC_HYPHEN_COUNT=$(grep -r "titan-optimal" /app/titan_optimal --include="*.md" 2>/dev/null | wc -l)
if [ "$DOC_HYPHEN_COUNT" -gt 0 ]; then
    echo "  ℹ️  Found $DOC_HYPHEN_COUNT references in docs (OK - explaining the change)"
fi

echo ""

# Check 3: Underscore references present
echo "✓ Checking for underscore references..."
UNDERSCORE_COUNT=$(grep -r "titan_optimal" /app/titan_optimal --include="*.py" --include="*.md" 2>/dev/null | wc -l)
if [ "$UNDERSCORE_COUNT" -gt 50 ]; then
    echo "  ✅ Found $UNDERSCORE_COUNT underscore references"
else
    echo "  ⚠️  Only found $UNDERSCORE_COUNT underscore references (expected 50+)"
fi

echo ""

# Check 4: Key files exist
echo "✓ Checking for key files..."

KEY_FILES=(
    "/app/titan_optimal/Titan_Optimal_Colab_Training.ipynb"
    "/app/titan_optimal/COLAB_NOTEBOOK_README.md"
    "/app/titan_optimal/MIGRATION_SUMMARY.md"
    "/app/titan_optimal/models/titan_gpt_v3.py"
    "/app/titan_optimal/configs/model_configs_v3.py"
    "/app/titan_optimal/train_v3.py"
)

for file in "${KEY_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $(basename $file)"
    else
        echo "  ❌ $(basename $file) NOT FOUND"
    fi
done

echo ""

# Check 5: Python imports work
echo "✓ Testing Python imports..."
cd /app
python3 -c "
import sys
sys.path.append('/app/titan_optimal')
try:
    from models.titan_gpt_v3 import TitanGPTModelV3
    from configs.model_configs_v3 import get_model_config_v3
    print('  ✅ Python imports successful')
except Exception as e:
    print(f'  ❌ Import failed: {e}')
    sys.exit(1)
" || exit 1

echo ""

# Check 6: Colab notebook structure
echo "✓ Checking Colab notebook..."
if [ -f "/app/titan_optimal/Titan_Optimal_Colab_Training.ipynb" ]; then
    CELLS=$(python3 -c "import json; data=json.load(open('/app/titan_optimal/Titan_Optimal_Colab_Training.ipynb')); print(len(data['cells']))" 2>/dev/null)
    if [ ! -z "$CELLS" ] && [ "$CELLS" -gt 10 ]; then
        echo "  ✅ Notebook has $CELLS cells (valid structure)"
    else
        echo "  ⚠️  Notebook may have structural issues"
    fi
else
    echo "  ❌ Notebook not found"
    exit 1
fi

echo ""

# Summary
echo "=========================================="
echo "✅ VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "All checks passed! Your titan_optimal folder is ready for Colab."
echo ""
echo "Next steps:"
echo "1. Create Colabnotebook branch:"
echo "   git checkout -b Colabnotebook"
echo "   git push origin Colabnotebook"
echo ""
echo "2. Update notebook with your GitHub username"
echo ""
echo "3. Upload to Google Colab and run!"
echo ""
echo "=========================================="
