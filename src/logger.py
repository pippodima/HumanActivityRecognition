import sys
import logbook


# Create a logger instance
script_logger = logbook.Logger("HAR-Logger")
# Add a handler to control where logs are sent (console, file, etc.)
handler = logbook.StreamHandler(sys.stdout, level=logbook.INFO)
handler.push_application()

