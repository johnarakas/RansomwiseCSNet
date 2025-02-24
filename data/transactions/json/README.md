# Transaction files (JSON)

Place JSON files containing transaction data in this directory.

The files' name should end in `.json` and the content should follow this format:

```json
{
    "1KtPALy1pCmsqpjbcJM5PRJfWw2UG2cKm6": {
        "in_timestamps": [1459786410], 
        "out_timestamps": [1462351285], 
        "incoming_transactions_btc": [0.5], 
        "incoming_transactions_euro": [185.933760058404], 
        "outcoming_transactions_btc": [0.5], 
        "outcoming_transactions_euro": [198.07681079809572]
        }, 
    "1K8iBHqgRFRFWVTrRZ7QNXvk8jhK3ZmFAv": {
        "in_timestamps": [1471658006, 1470988229, 1470381027], 
        "out_timestamps": [1471682778, 1471013243, 1470384394], 
        "incoming_transactions_btc": [1.2, 1.2, 1.2], 
        "incoming_transactions_euro": [629.344175730444, 644.7742292971221, 661.609645577667], "outcoming_transactions_btc": [1.2, 1.2, 1.2], 
        "outcoming_transactions_euro": [629.344175730444, 644.7742292971221, 661.609645577667]
        }, 
    ... 
}
```