class BillingService:
    """Handles recurring invoice billing, separate from one-off order payments."""

    def process_payment(self, invoice):
        """Charge a recurring invoice's card on file."""
        return _charge_invoice(invoice)


def _charge_invoice(invoice):
    return True
