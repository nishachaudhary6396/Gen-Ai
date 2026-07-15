from langchain_community.document_loaders import WebBaseLoader

from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI()

prompt = PromptTemplate(
    template='Answer the following question - \n  {question} from the following text -\n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

loader = WebBaseLoader("https://www.flipkart.com/timex-twhg03smu19-multi-function-black-dial-analog-watch-men/p/itm453230224a2d3?pid=WATGFSRQQHVK455G&lid=LSTWATGFSRQQHVK455GSQF63O&marketplace=FLIPKART&store=r18%2Ff13&spotlightTagId=default_BestsellerId_r18%2Ff13&srno=b_1_4&otracker=browse&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_6_L1_view-all&fm=organic&iid=d8e9d92f-278f-4557-baad-8256997116a6.WATGFSRQQHVK455G.SEARCH&ppt=browse&ppn=browse&ov_redirect=true")

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question': 'What is the price of the watch?', 'text': docs[0].page_content}))
