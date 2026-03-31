#!/bin/bash
# Release Script for AI-Mastery-2026
# ===================================
# Automates the release process including version bumping, changelog updates,
# and git tagging.
#
# Usage:
#   ./scripts/release.sh <version>     # Release new version (e.g., 0.2.0)
#   ./scripts/release.sh --dry-run     # Preview changes without committing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DRY_RUN=false
VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

# Validate version
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version required${NC}"
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 0.2.0"
    exit 1
fi

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version must be in semver format: MAJOR.MINOR.PATCH"
    echo "Example: 0.2.0"
    exit 1
fi

echo "=================================="
echo "AI-Mastery-2026 Release Script"
echo "=================================="
echo -e "${BLUE}Version:${NC} $VERSION"
echo -e "${BLUE}Dry Run:${NC} $DRY_RUN"
echo ""

# Check current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: Not on main branch (current: $BRANCH)${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Uncommitted changes detected${NC}"
    echo "Please commit or stash changes before releasing"
    exit 1
fi

echo -e "${YELLOW}Step 1: Updating version in pyproject.toml...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would update version to $VERSION"
else
    # Update version in pyproject.toml
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    rm pyproject.toml.bak
    echo -e "${GREEN}✓ Updated pyproject.toml${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Updating CHANGELOG.md...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would update CHANGELOG.md with release date"
else
    # Update CHANGELOG with release date
    DATE=$(date +%Y-%m-%d)
    sed -i.bak "s/## \[Unreleased\]/## [Unreleased]\n\n---\n\n## [$VERSION] - $DATE/" CHANGELOG.md
    rm CHANGELOG.md.bak
    echo -e "${GREEN}✓ Updated CHANGELOG.md${NC}"
fi

echo ""
echo -e "${YELLOW}Step 3: Committing changes...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would commit with message: release: v$VERSION"
else
    git add pyproject.toml CHANGELOG.md
    git commit -m "release: v$VERSION"
    echo -e "${GREEN}✓ Committed changes${NC}"
fi

echo ""
echo -e "${YELLOW}Step 4: Creating git tag...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create tag v$VERSION"
else
    git tag -a "v$VERSION" -m "Release v$VERSION"
    echo -e "${GREEN}✓ Created tag v$VERSION${NC}"
fi

echo ""
echo -e "${YELLOW}Step 5: Pushing to remote...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would push to origin main and tag v$VERSION"
else
    git push origin main
    git push origin "v$VERSION"
    echo -e "${GREEN}✓ Pushed to remote${NC}"
fi

echo ""
echo "=================================="
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo "No changes were made"
else
    echo -e "${GREEN}RELEASE COMPLETE!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Create GitHub release: https://github.com/Kandil7/AI-Mastery-2026/releases/new"
    echo "  2. Tag: v$VERSION"
    echo "  3. CD workflow will automatically publish to PyPI"
fi
echo "=================================="
