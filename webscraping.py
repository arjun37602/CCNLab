import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

users = []
dates = []
goals = []
contract_id = []
category = []
#mimics browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

url = "https://www.stickk.com/communities/index?type=weight"
response = requests.get(url, headers=headers)
html = BeautifulSoup(response.content, 'html.parser')
entries = html.find_all('div', class_='communitiesStickkLiveTextContainer')
# extract features from raw HTML
#<a class="communitiesCommunityJournalUsername" href="/commitment/601182">
for entry in entries:
    dates.append(entry.find('span', class_="communitiesStickkLiveTime"))
    users.append(entry.find('a', class_="communitiesCommunityJournalUsername"))
    goals.append(entry.find('span', class_="communityJournalJournal"))
    contract_id.append(entry.find('a', class_='communitiesCommunityJournalUsername')['href'])
    category.append('Weight')

# for i, user in enumerate(users):
#     if "\n" in user:
#         users[i] = user.replace("\n", "")
for i, date in enumerate(dates):
    dates[i] = str(date)
    if 'span' in date:
        pat = r">([^<]+)</span>"
        dates[i] = re.findall(pat, date)
for i, user in enumerate(users):
    users[i] = str(users)
    if 'a' in date:
        pat = r">([^<]+)</a>"
        users[i] = re.findall(pat, user)
for i, goal in enumerate(goals):
    goals[i] = str(goal)
    if 'span' in goal:
        pat = r">([^<]+)</span>"
        goals[i] = re.findall(pat, goal)
pattern = r"commitment/(\d+)"
for j, i in enumerate(contract_id):
    i = str(i)
    if 'commitment' in i:
        temp = re.findall(pattern, i)
        contract_id[j] = temp[0]
    else:
        contract_id[j] = i

df = pd.DataFrame({'Date': dates, 'Users': users, 'Single Entry': goals, 'Category': category, 'Contract_id': contract_id})

def extract_span(word):
    word = str(word)
    pat = r">([^<]+)</span>"
    return re.findall(pat, word)[0]
def extract_div(word):
    word = str(word)
    pat = r">([^<]+)</div>"
    return re.findall(pat, word)[0]

contract_goals = {}
details = {}
counter = 0
for contract in contract_id:
    target_url = 'https://www.stickk.com/commitment/details/' + str(contract) + ''
    target_response = requests.get(target_url, headers=headers)
    target_html = BeautifulSoup(target_response.content, 'html.parser')
    target_entries = target_html.find_all('div', id="commitmentSummaryICommitToText")
    # label_entries = target_html.find_all('span', class_="label")
    # cell_entries = target_html.find_all('span', class_="cell2")
    for tags in target_entries:
        val = tags.find('span')
        val = extract_span(val)
        contract_goals[str(contract)] = val
print(contract_goals)
# ultimate_goal = []
# target_weight = []
# weekly_loss_pace = []
# actual_loss_pace = []
# current_weight = []
# on_paceto_weight = []
# target.append(target_entries.find('span',id="commitmentTitle"))
# print(target_entries.find('span', class_="cell2"))

