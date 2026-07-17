def send_notification(user, message):
    """Send a notification to a user."""
    return _dispatch(user, message)


def _internal_helper(x):
    # Purely internal, never referenced by any doc — should never get
    # a false link no matter what else is in the docs.
    return x * 2


def _dispatch(user, message):
    return True
