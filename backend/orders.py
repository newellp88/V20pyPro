# class to support opening and closing trades
import json
from .config import accountID, token
from oandapyV20 import API
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import TradeCloseRequest
from oandapyV20.exceptions import V20Error
from .account import margin_available, margin_rate
from oandapyV20.contrib.requests import (MarketOrderRequest,
                                         TakeProfitDetails,
                                         StopLossDetails,
                                         TrailingStopLossOrderRequest)

api = API(token)

def weight(price, allocation):
    trade_margin = float(margin_available) * allocation
    trade_value = trade_margin / float(margin_rate)
    w = round((trade_value/float(price)), 0)
    return w

def order(instrument, weight, price, TP, SL):

    if weight > 0:
        mktOrder = MarketOrderRequest(
                        instrument=instrument,
                        units=weight,
                        takeProfitOnFill=TakeProfitDetails(price=TP).data,
                        stopLossOnFill=StopLossDetails(price=SL).data)
        r = orders.OrderCreate(accountID, data=mktOrder.data)

        try:
            rv = api.request(r)
            #print(r.status_code)
        except V20Error as e:
            print(r.status_code, e)
        else:
            #print(json.dumps(rv, indent=4))
            pass

    else:
        mktOrder = MarketOrderRequest(
                            instrument=instrument,
                            units=weight,
                            takeProfitOnFill=TakeProfitDetails(price=TP).data,
                            stopLossOnFill=StopLossDetails(price=SL).data)
        r = orders.OrderCreate(accountID, data=mktOrder.data)

        try:
            rv = api.request(r)
            #print(r.status_code)
        except V20Error as e:
            print(r.status_code, e)
        else:
            #print(json.dumps(rv, indent=4))
            pass

    return r.response

def trailingStop(tradeID, distance=0.1):
    activeStop = TrailingStopLossOrderRequest(tradeID=tradeID,
                    r = orders.OrderCreate(accountID,
                    data=activeStop.data))
    api.request(r)
    return r.response

def close(tradeID):
    close_trade = TradeCloseRequest()
    r = trades.TradeClose(accountID, tradeID, data=close_trade.data)
    api.request(r)
    return r.response

def closeAll(tradeID_list):
    for trade in tradeID_list:
        close(trade)
        print(trade, "Closed")
