# Billing

## Payments
Both order checkout and recurring invoices call `process_payment`
internally to charge a card.

## Refunds
Call `refund(order_id)` to refund a previously processed order.
