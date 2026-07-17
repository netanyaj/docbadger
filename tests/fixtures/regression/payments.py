class PaymentProcessor:
    """Handles one-off payment processing for orders."""

    def process_payment(self, order):
        """Charge the order's card."""
        return _charge_card(order)

    def refund(self, order_id):
        """Refund a previously processed order."""
        return _issue_refund(order_id)


def _charge_card(order):
    return True


def _issue_refund(order_id):
    return True
