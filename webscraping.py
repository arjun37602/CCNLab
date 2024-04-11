import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

url = "https://www.stickk.com/communities/index?type=weight"

#mimics browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}


response = requests.get(url, headers=headers)
html = BeautifulSoup(response.content, 'html.parser')
entries = html.find_all('div', class_='communitiesStickkLiveTextContainer')
users = []
dates = []
goals = []
contract_id = []
# extract features from raw HTML
#<a class="communitiesCommunityJournalUsername" href="/commitment/601182">
for entry in entries:
    dates.append(entry.find('span', class_="communitiesStickkLiveTime"))
    users.append(entry.find('a', class_="communitiesCommunityJournalUsername"))
    goals.append(entry.find('span', class_="communityJournalJournal"))
    contract_id.append(entry.find('a', class_='communitiesCommunityJournalUsername')['href'])
pattern = r"commitment/(\d+)"
for j, i in enumerate(contract_id):
    temp = re.findall(pattern, i)
    contract_id[j] = temp
print(contract_id)
df = pd.DataFrame({'Date': dates, 'Users': users, 'Goals': goals})
print(df)
