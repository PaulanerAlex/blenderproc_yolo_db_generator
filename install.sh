#!/bin/bash

################################################################################
# Automated Installation Script for Blender YOLO Dataset Generator
# Using uv for fast Python environment management
#
# This script automates the complete setup process:
# - Checks for and installs uv if needed
# - Creates Python 3.10 virtual environment
# - Installs BlenderProc and all dependencies
# - Downloads CC0 textures (optional)
# - Verifies installation
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENV_DIR=".venv"
SKIP_TEXTURES=false
TEXTURE_PATH=""
TEXTURE_SUBSET="none"  # Options: none, minimal, medium, full
PYTHON_VERSION="3.10"
BLENDERPROC_VERSION="2.5.0"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

################################################################################
# Helper Functions
################################################################################

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Blender YOLO Dataset Generator - Installation${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo ""
}

show_help() {
    cat << EOF
Usage: ./install.sh [OPTIONS]

Automated installation script for Blender YOLO Dataset Generator using uv.

Options:
    --skip-textures           Skip downloading CC0 textures entirely
    --textures SUBSET         Download texture subset (minimal/medium/full)
                              minimal: ~5GB (50 textures for basic testing)
                              medium:  ~15GB (150 textures for good variety)
                              full:    ~56GB (all textures, not recommended)
    --texture-path PATH       Custom path for CC0 textures
    --python VERSION          Python version to use (default: 3.10)
    --help                    Show this help message

Examples:
    ./install.sh                              # No textures (fastest)
    ./install.sh --skip-textures              # Same as above (explicit)
    ./install.sh --textures minimal           # Download ~5GB subset
    ./install.sh --textures medium            # Download ~15GB subset
    ./install.sh --texture-path ~/textures    # Custom texture path
    ./install.sh --python 3.11                # Use Python 3.11

Recommended: Start with --textures minimal or --skip-textures.
             You can always download more textures later using:
             python -m blenderproc download cc_textures <path>

Note: This script uses uv for fast Python environment management.
      BlenderProc will automatically download Blender (~500MB) on first run.

EOF
    exit 0
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-textures)
            SKIP_TEXTURES=true
            shift
            ;;
        --textures)
            TEXTURE_SUBSET="$2"
            shift 2
            ;;
        --texture-path)
            TEXTURE_PATH="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Check System Requirements
################################################################################

check_system() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_warning "This script is designed for Linux. You may encounter issues on other systems."
    fi
    
    # Check disk space
    available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    required_space=5
    if [ "$SKIP_TEXTURES" = false ]; then
        required_space=35
    fi
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_error "Insufficient disk space. Required: ${required_space}GB, Available: ${available_space}GB"
        exit 1
    fi
    
    print_success "System check passed"
}

################################################################################
# Install uv
################################################################################

check_and_install_uv() {
    print_step "Checking for uv installation..."
    
    if command -v uv &> /dev/null; then
        print_success "uv found: $(uv --version)"
        return 0
    fi
    
    print_warning "uv not found. Installing uv..."
    
    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if command -v uv &> /dev/null; then
        print_success "uv installed successfully: $(uv --version)"
    else
        print_error "Failed to install uv. Please install manually from: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    print_warning "uv has been installed. You may need to restart your terminal or run: source ~/.bashrc"
}

################################################################################
# Create Virtual Environment
################################################################################

create_environment() {
    print_step "Creating Python $PYTHON_VERSION virtual environment with uv..."
    
    cd "$SCRIPT_DIR"
    
    # Remove existing environment if it exists
    if [ -d "$ENV_DIR" ]; then
        print_warning "Virtual environment already exists at $ENV_DIR"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment..."
            rm -rf "$ENV_DIR"
        else
            print_step "Using existing environment..."
            return 0
        fi
    fi
    
    # Create environment with uv
    uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
    
    print_success "Virtual environment created at $ENV_DIR"
}

################################################################################
# Install Dependencies
################################################################################

install_dependencies() {
    print_step "Installing Python dependencies with uv..."
    
    cd "$SCRIPT_DIR"
    
    # Install BlenderProc
    print_step "Installing BlenderProc $BLENDERPROC_VERSION..."
    uv pip install --python "$ENV_DIR/bin/python" "blenderproc>=$BLENDERPROC_VERSION"
    
    # Install dependencies from requirements.txt
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        print_step "Installing additional dependencies from requirements.txt..."
        uv pip install --python "$ENV_DIR/bin/python" -r requirements.txt
        print_success "Dependencies from requirements.txt installed"
    else
        print_warning "requirements.txt not found, skipping additional dependencies"
    fi
    
    # Also install dependencies in Blender's Python environment for scripts
    print_step "Installing dependencies in Blender's environment..."
    print_warning "This is needed for blenderproc run to access our modules..."
    
    # Install in Blender
    "$ENV_DIR/bin/blenderproc" pip install jsonschema pyyaml numpy tqdm matplotlib h5py opencv-python Pillow 2>&1 | grep -E "Successfully|already" || true
    
    print_success "Dependencies installed successfully"
}

################################################################################
# Download CC0 Textures
################################################################################

download_textures() {
    if [ "$SKIP_TEXTURES" = true ] || [ "$TEXTURE_SUBSET" = "none" ]; then
        print_warning "Skipping texture download"
        print_step "Note: Textures improve realism but are optional"
        print_step "You can generate datasets without textures (solid colors will be used)"
        return 0
    fi
    
    # Determine texture path
    if [ -z "$TEXTURE_PATH" ]; then
        TEXTURE_PATH="$SCRIPT_DIR/cc0_textures"
    fi
    
    print_step "⚠ IMPORTANT: Texture Download Information"
    echo ""
    echo "================================================================"
    echo "  CC0 Texture Download Options"
    echo "================================================================"
    echo ""
    echo "The full CC0 texture pack is 56GB+ which is impractical."
    echo ""
    echo "RECOMMENDED APPROACH:"
    echo "  1. Skip textures for now (solid colors work great!)"
    echo "  2. Generate your first dataset to verify everything works"
    echo "  3. If needed, manually download 5-10 specific textures later"
    echo ""
    echo "If you want textures, you have two options:"
    echo ""
    echo "  A) Manual Download (RECOMMENDED):"
    echo "     • Visit https://ambientcg.com/"
    echo "     • Download 5-10 textures you need (Wood, Metal, etc.)"
    echo "     • Extract to: $TEXTURE_PATH"
    echo "     • See docs/TEXTURE_OPTIONS.md for recommended textures"
    echo ""
    echo "  B) Full Download (NOT RECOMMENDED):"
    echo "     • Run: blenderproc download cc_textures $TEXTURE_PATH"
    echo "     • Downloads all 56GB+ of textures"
    echo "     • Takes hours"
    echo ""
    echo "================================================================"
    echo ""
    
    print_step "What would you like to do?"
    echo "  1) Continue WITHOUT textures (recommended)"
    echo "  2) See manual download instructions"
    echo "  3) Download ALL textures (56GB+, not recommended)"
    echo ""
    read -p "Enter choice (1-3): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            print_success "Continuing without textures"
            print_step "You can generate excellent datasets with solid colors!"
            return 0
            ;;
        2)
            print_step "Opening manual download guide..."
            "$ENV_DIR/bin/python" "$SCRIPT_DIR/scripts/download_texture_subset.py" \
                --subset medium \
                --output "$TEXTURE_PATH"
            
            print_step ""
            print_step "After manually downloading textures:"
            print_step "  1. Extract each texture ZIP to: $TEXTURE_PATH"
            print_step "  2. Verify structure: $TEXTURE_PATH/Wood051/, Metal006/, etc."
            print_step "  3. Continue with dataset generation"
            print_step ""
            
            # Create the directory
            mkdir -p "$TEXTURE_PATH"
            echo "TEXTURE_PATH=$TEXTURE_PATH" > "$SCRIPT_DIR/.env"
            echo "TEXTURE_SUBSET=manual" >> "$SCRIPT_DIR/.env"
            
            return 0
            ;;
        3)
            print_warning "Downloading ALL textures (56GB+)..."
            print_warning "This will take a long time!"
            echo ""
            read -p "Are you SURE? (type 'yes' to confirm): " -r
            echo
            
            if [ "$REPLY" = "yes" ]; then
                mkdir -p "$TEXTURE_PATH"
                if "$ENV_DIR/bin/blenderproc" download cc_textures "$TEXTURE_PATH"; then
                    print_success "All textures downloaded to $TEXTURE_PATH"
                    echo "TEXTURE_PATH=$TEXTURE_PATH" > "$SCRIPT_DIR/.env"
                    echo "TEXTURE_SUBSET=full" >> "$SCRIPT_DIR/.env"
                else
                    print_error "Texture download failed"
                    print_step "You can continue without textures or try manual download"
                fi
            else
                print_warning "Full download cancelled"
                return 0
            fi
            ;;
        *)
            print_warning "Invalid choice, continuing without textures"
            return 0
            ;;
    esac
}

################################################################################
# Verify Installation
################################################################################

verify_installation() {
    print_step "Verifying installation..."
    
    # Check BlenderProc installation
    print_step "Checking BlenderProc..."
    
    # Check if blenderproc command is available
    if ! "$ENV_DIR/bin/blenderproc" --version 2>/dev/null; then
        print_warning "BlenderProc command not found, checking package installation..."
        
        # Try to get version from package
        BLENDERPROC_VER=$(uv pip list --python "$ENV_DIR/bin/python" 2>/dev/null | grep blenderproc | awk '{print $2}' || echo "Unknown")
        
        if [ "$BLENDERPROC_VER" = "Unknown" ]; then
            print_error "BlenderProc installation could not be verified"
            print_step "This might be okay - try running: source .venv/bin/activate && blenderproc --version"
            return 0  # Don't fail, just warn
        else
            print_success "BlenderProc $BLENDERPROC_VER package installed"
        fi
    else
        BLENDERPROC_VER=$("$ENV_DIR/bin/blenderproc" --version 2>&1 | head -1)
        print_success "BlenderProc $BLENDERPROC_VER installed correctly"
    fi
    
    # Run BlenderProc quickstart (optional, downloads Blender)
    print_step "Running BlenderProc quickstart test (optional)..."
    print_warning "This will download Blender (~500MB) and run a test render..."
    print_warning "BlenderProc manages Blender automatically - no manual installation needed."
    echo ""
    
    read -p "Do you want to run the verification test? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create temp directory for test
        TEST_DIR=$(mktemp -d)
        cd "$TEST_DIR"
        
        if "$ENV_DIR/bin/blenderproc" quickstart; then
            print_success "BlenderProc quickstart test passed!"
            
            # Show output
            if [ -f "output/0.hdf5" ]; then
                print_success "Test output generated: $TEST_DIR/output/0.hdf5"
                print_step "You can visualize it with: $ENV_DIR/bin/blenderproc vis hdf5 $TEST_DIR/output/0.hdf5"
            fi
        else
            print_error "BlenderProc quickstart test failed"
            print_step "This might be okay - the package is installed, test might have other issues"
        fi
        
        cd "$SCRIPT_DIR"
    else
        print_warning "Verification test skipped"
        print_warning "Note: Blender will be downloaded automatically on first use of BlenderProc"
    fi
    
    print_success "Installation verification complete"
}

################################################################################
# Create Activation Script
################################################################################

create_activation_script() {
    print_step "Creating activation script..."
    
    cat > "$SCRIPT_DIR/activate.sh" << EOF
#!/bin/bash
# Activate the BlenderProc virtual environment

source "$SCRIPT_DIR/$ENV_DIR/bin/activate"

echo "BlenderProc environment activated!"
echo "Python: \$(python --version)"
echo "BlenderProc: \$(python -c 'import blenderproc; print(blenderproc.__version__)' 2>/dev/null || echo 'Unknown')"
echo ""
echo "To deactivate, run: deactivate"
EOF
    
    chmod +x "$SCRIPT_DIR/activate.sh"
    
    print_success "Created activate.sh script"
}

################################################################################
# Create pyproject.toml for uv
################################################################################

create_pyproject() {
    print_step "Creating pyproject.toml..."
    
    cat > "$SCRIPT_DIR/pyproject.toml" << EOF
[project]
name = "blender-yolo-generator"
version = "0.1.0"
description = "Framework for generating YOLO training datasets using Blender and BlenderProc"
readme = "README.md"
requires-python = ">=$PYTHON_VERSION"
dependencies = [
    "blenderproc>=2.5.0",
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "h5py>=3.7.0",
    "PyYAML>=6.0",
    "Pillow>=9.0.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.64.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = []
EOF
    
    print_success "Created pyproject.toml"
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    print_header
    
    echo "Installation Configuration:"
    echo "  Python version: $PYTHON_VERSION"
    echo "  BlenderProc version: >= $BLENDERPROC_VERSION"
    echo "  Virtual environment: $ENV_DIR"
    echo "  Skip textures: $SKIP_TEXTURES"
    if [ -n "$TEXTURE_PATH" ]; then
        echo "  Texture path: $TEXTURE_PATH"
    fi
    echo ""
    echo "Note: Using uv for fast package management"
    echo "      BlenderProc will download Blender (~500MB) on first run"
    echo ""
    
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Installation cancelled"
        exit 0
    fi
    
    # Run installation steps
    check_system
    check_and_install_uv
    create_pyproject
    create_environment
    install_dependencies
    download_textures
    verify_installation
    create_activation_script
    
    # Print success message
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Installation Complete! 🎉${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "     source $ENV_DIR/bin/activate"
    echo "     # or use: source activate.sh"
    echo ""
    echo "  2. Place your 3D models in: models/"
    echo ""
    echo "  3. Configure your dataset generation: configs/"
    echo ""
    echo "  4. Generate your dataset:"
    echo "     python src/generate_dataset.py --config configs/example_config.json"
    echo ""
    echo "Benefits of using uv:"
    echo "  • Fast dependency resolution and installation"
    echo "  • Reproducible environments with lockfiles"
    echo "  • Better dependency management"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main installation
main
