import sys
import os

# Add the current directory to sys.path so that imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    import main
    print("Successfully imported main. Running main()...")
    main.main()
    print("Finished running main()")
except Exception as e:
    import traceback
    traceback.print_exc()
