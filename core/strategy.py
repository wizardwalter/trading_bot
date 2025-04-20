import random

def should_enter_trade(ticker):
    # Placeholder for actual ML/indicator logic
    enter = random.choice([True, False])
    price = round(random.uniform(100, 400), 2)
    return {"enter": enter, "price": price}