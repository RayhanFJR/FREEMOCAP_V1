"""
Main entry point untuk aplikasi FreeMoCap
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main function dengan pilihan menu"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1]
        
        if command == 'visualize' or command == 'viz':
            # Buka visualizer GUI
            from visualization.visualize_gui import main as viz_main
            viz_main()
        elif command == 'capture':
            # Buka aplikasi capture
            from gui.ui_main import main as ui_main
            ui_main()
        else:
            print("Usage:")
            print("  python main.py              : Buka aplikasi capture")
            print("  python main.py capture      : Buka aplikasi capture")
            print("  python main.py visualize    : Buka visualizer")
            print("  python main.py viz          : Buka visualizer (shortcut)")
    else:
        # Default: buka aplikasi capture
        from gui.ui_main import main as ui_main
        ui_main()

if __name__ == '__main__':
    main()
