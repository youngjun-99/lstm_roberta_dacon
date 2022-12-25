from tqdm import tqdm
import collections
import pandas as pd
import os
import torch
import random
import numpy as np

from sklearn.metrics import f1_score

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

import time
import pyperclip

def set_allseed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(pred, num_labels):
    predict = pred.predictions.argmax(axis=1)
    ref = pred.label_ids
    pred_li, ref_li = [], []
    for i, j in zip (predict, ref):
        prediction, reference = [0] * num_labels, [0] * num_labels
        prediction[i] = 1
        reference[j] = 1
        pred_li.append(prediction)
        ref_li.append(reference)
    f1 = f1_score(pred_li, ref_li, average="weighted")
    return {'f1' : f1 }

def translate(text_data, data_lang, trans_lang):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver', options=chrome_options)

    target = EC.presence_of_element_located((By.XPATH, '#txtTarget'))
    input = EC.presence_of_element_located((By.CSS_SELECTOR, '#txtSource'))

    trans_list = []

    for i in tqdm(range(len(text_data))):
        counter = 0
        driver.get('https://papago.naver.com/?sk='+ data_lang +'&tk='+trans_lang)
        try:
            pyperclip.copy(text_data[i])
            WebDriverWait(driver, 3).until(input).click()
            WebDriverWait(driver, 3).until(input).send_keys(Keys.CONTROL, "v")

            while True:
                if (backtrans=='')|(backtrans==' ')|('...' in backtrans)|(counter <= 10):
                    backtrans = WebDriverWait(driver, 3).until(target).text
                    time.sleep(1)
                else:
                    trans_list.append(backtrans)
                    break
            
        except:
            trans_list.append('')
    
    return pd.DataFrame(trans_list)