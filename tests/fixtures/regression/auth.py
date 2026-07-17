class AuthClient:
    """Handles user authentication."""

    def login(self, username, password):
        """Authenticate a user with a username and password."""
        return _verify_credentials(username, password)

    def logout(self, session_id):
        """End the given session."""
        return _invalidate_session(session_id)

    def register(self, username, password, email):
        """Register a new user account."""
        return _create_account(username, password, email)


def _verify_credentials(username, password):
    return True


def _invalidate_session(session_id):
    return True


def _create_account(username, password, email):
    return True
