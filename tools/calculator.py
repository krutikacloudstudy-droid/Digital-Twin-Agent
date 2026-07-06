def calculator(query):
    try:
        return str(eval(query))
    except:
        return "Invalid calculation"