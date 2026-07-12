def send_email(to, subject, body):
    """Send an email. Does not retry on failure."""
    smtp_client.send(to, subject, body)
