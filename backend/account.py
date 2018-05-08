# account details for export

from oandapyV20 import API
from oandapyV20.endpoints.accounts import AccountSummary
from .config import accountID, token

api = API(token)

r = AccountSummary(accountID)
summary = api.request(r) # base dict
account = summary['account'] # item containing dict of account data

account_summary = dict()

balance = account['balance']
open_trades = account['openTradeCount']
open_positions = account['openPositionCount']
pnl = account['pl']
carrying_costs = account['financing']
open_pnl = account['unrealizedPL']
nav = account['NAV']
margin_used = account['marginUsed']
margin_available = account['marginAvailable']
margin_call_pct = account['marginCallPercent']
margin_rate = account['marginRate']
