import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import requests
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent


st.sidebar.title("Add API Keys Below:")
alphav_api = st.sidebar.text_input("Alpha Vantage API Key", type="password")
openai_api = st.sidebar.text_input("OpenAI API Key", type="password")
fm_api = st.sidebar.text_input("Financial Modeling Prep API Key", type="password")


col1, col2 = st.columns(2)
with col1:
    with open("htnlogo.svg", "r") as file:
        svg_logo = file.read()
    st.markdown(svg_logo, unsafe_allow_html=True)

def get_current_stock_ratio(ticker):
    """Method to get a stocks fundamental or ratio information"""
    url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={fm_api}"
    
    return requests.get(url).json()
# base model
class CurrentStockRatioInput(BaseModel):
    """Inputs for get_current_stock_ratio"""
    ticker: str = Field(description="Ticker symbol of the stock")
# base tool
class CurrentStockRatioTool(BaseTool):
    name = "get_current_stock_ratio"
    description = """
        Useful when you want to get current stock fundamentals and ratios.
        You should enter the stock ticker symbol recognized stock exchanges
        """
    args_schema: Type[BaseModel] = CurrentStockRatioInput
    def _run(self, ticker: str):
        ratio_response = get_current_stock_ratio(ticker)
        return ratio_response
    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_ratio does not support async")
    
# sell side analyst api
def get_analyst_esti(ticker):
    """Method to get sell-side analyst estimates on a stock """  
    url = f"https://financialmodelingprep.com/api/v3/grade/{ticker}?limit=5&apikey={fm_api}"
    return requests.get(url).json()
class CurrentAnalystInput(BaseModel):
    """Inputs for get_analyst_esti"""
    ticker: str = Field(description="Ticker symbol of the stock")
# base tool
class CurrentAnalystTool(BaseTool):
    name = "get_analyst_esti"
    description = """
        Useful for when the user wants to get bank or analyst opinions or estimates on a stock
        """
    args_schema: Type[BaseModel] = CurrentAnalystInput
    def _run(self, ticker: str):
        analyst_response = get_analyst_esti(ticker)
        return analyst_response
    def _arun(self, ticker: str):
        raise NotImplementedError("get_analyst_esti does not support async")
    


def get_revenue_seg(ticker):
    """Method to get product revenue segementation of a company"""  
    url = f"https://financialmodelingprep.com/api/v4/revenue-product-segmentation?symbol={ticker}?limit=5&structure=flat&apikey={fm_api}"
    return requests.get(url).json()
class CurrentRevenueSegInput(BaseModel):
    """Inputs for get_revenue_seg"""
    ticker: str = Field(description="Ticker symbol of the stock")
# base tool
class CurrentRevenueSegTool(BaseTool):
    name = "get_revenue_seg"
    description = """
        Useful for when the user wants to get bank or analyst opinions or estimates on a stock
        """
    args_schema: Type[BaseModel] = CurrentRevenueSegInput
    def _run(self, ticker: str):
        revenue_response = get_revenue_seg(ticker)
        return revenue_response
    def _arun(self, ticker: str):
        raise NotImplementedError("get_revenue_seg does not support async")


def get_alpha_vantage_commodity(commodity):
    """Method to get information on commoditys such as Crude oil which has a symbol of WTI, Brent, NATURAL_GAS, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, and Coffee. Those listed commodities can be passed into the url as commodity"""  
    url = f"https://www.alphavantage.co/query?function={commodity}&interval=daily&apikey={alphav_api}"
    return requests.get(url).json()
class CommodityInput(BaseModel):
    """Inputs for get_alpha_vantage_commodity"""
    commodity: str = Field(description="Symbol of the commodity")
# base tool
class CommodityTool(BaseTool):
    name = "get_alpha_vantage_commodity"
    description = """
        Useful for when a user wants to know anything about commodities listed: WTI, Brent, NATURAL_GAS, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, and Coffee
        """
    args_schema: Type[BaseModel] = CommodityInput
    def _run(self, commodity: str):
        commodity_response = get_alpha_vantage_commodity(commodity)
        return commodity_response
    def _arun(self, commodity: str):
        raise NotImplementedError("get_alpha_vantage_commodity does not support async")


def get_gain_lose():
    url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alphav_api}"
    return requests.get(url).json()

class GainLoseInput(BaseModel):
    pass

class GainLoseTool(BaseTool):
    name = "get_gain_lose"
    description = "Useful for when a user wants to know which stocks lost and gained the most"
    args_schema: Type[BaseModel] = GainLoseInput

    def _run(self):
        response = get_gain_lose()
        return response

    def _arun(self):
        raise NotImplementedError("get_gain_lose does not support async")

# # ai logic
# llm = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key = "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", temperature=0) 
# tools = [CurrentStockRatioTool(), CurrentAnalystTool(), CurrentRevenueSegTool(), CommodityTool(), GainLoseTool()]
# agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# agent.run(
#     question
# )

question = st.text_input('Please enter your question:')
if question:
    st.write(f'You asked: {question}')
    
    with st.spinner('Loading your answer...'):
        # ai logic
        llm = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key = openai_api, temperature=0) # the temp of 0 makes it less creative and more based in the pure facts it sees in the api
        tools = [CurrentStockRatioTool(), CurrentAnalystTool(), CurrentRevenueSegTool(), CommodityTool(), GainLoseTool()]
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

        response = agent.run(question)
        # puts response text into a dataframe
        response_df = pd.DataFrame([response])
        csv_data = response_df.to_csv(index=False)
        
        st.write('Your answer:', response)
        # download button
        st.download_button(
            label="Download answer as CSV",
            data=csv_data,
            file_name='answer.csv',
            mime='text/csv',
        )


