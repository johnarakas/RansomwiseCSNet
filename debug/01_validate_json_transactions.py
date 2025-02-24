json_file = "data/transactions/json/ransomwareTransactions.json"

def count_wallet_number(transactions: dict):
    """
    Args:
        transactions (dict): Transaction dictionary with wallet addresses (str) as keys and transactions histories (dict) as values
    
    Return:
        wallet_count (int): Number of wallets in the transaction dictionary
    """

    assert isinstance(list(transactions.keys())[0], str), "Error: dict keys are not wallet addresses" 

    return len(transactions)


if __name__ == "__main__":

    import json

    with open(json_file, 'r') as f:

        transactions = json.load(f)

    print("Number of wallets:", count_wallet_number(transactions))

    print(list(transactions.values())[0])