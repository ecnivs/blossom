import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path.resolve()))

if __name__ == "__main__":
    try:
        from main import main

        main()
    except ImportError as e:
        print(f"Failed to start application: {e}")
        print(f"Ensure that {src_path} exists and contains main.py")
        sys.exit(1)
