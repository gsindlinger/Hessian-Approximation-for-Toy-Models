import logging

# Prevent “No handler found” warnings for library users
logging.getLogger(__name__).addHandler(logging.NullHandler())
