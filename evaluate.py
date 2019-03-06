## code adapted from https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py

import json
import re

##load files
qns = json.load(open("v2_OpenEnded_mscoco_val2014_questions.json",'r'))
anns = json.load(open("v2_mscoco_val2014_annotations.json",'r'))
generated = json.load(open('results.json', 'r'))


###parse annotation and question jsons
questions = {}
target = {}
for q in qns["questions"]:
    questions[q["question_id"]] = q["question"] 
for a in anns["annotations"]:
    target[a["question_id"]] = a


##cleaning
punct = [';', "/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
manualMap = { 'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10'}
articles     = ['a','an','the']
periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip   = re.compile("(\d)(\,)(\d)")
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                "youll": "you'll", "youre": "you're", "youve": "you've"}


##functions to clean
def processPunctuation(inText):
		outText = inText
		for p in punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')	
		outText = periodStrip.sub("",outText,re.UNICODE)
		return outText

def processDigitArticle(inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = manualMap.setdefault(word, word)
			if word not in articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in contractions: 
				outText[wordId] = contractions[word]
		outText = ' '.join(outText)
		return outText


### evaluation function to calculate acc
def evaluate(quesIds, target, generated):
    accQA       = []
    accQuesType = {}
    accAnsType  = {}
    for quesId in quesIds:
        resAns      = generated[quesId]
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = processPunctuation(resAns)
        resAns      = processDigitArticle(resAns)
        gtAcc  = []
        quesId = int(quesId)
        gtAnswers = [str(ans['answer']) for ans in target[quesId]['answers']]
        if len(set(gtAnswers)) > 1: 
            for ansDic in target[quesId]['answers']:
                ansDic['answer'] = processPunctuation(ansDic['answer'])
        for gtAnsDatum in target[quesId]['answers']:
            otherGTAns = [item for item in target[quesId]['answers'] if item!=gtAnsDatum]
            matchingAns = [item for item in otherGTAns if item['answer']==resAns]
            acc = min(1, float(len(matchingAns))/3)
            gtAcc.append(acc)
        quesType = target[quesId]['question_type']
        ansType = target[quesId]['answer_type']
        avgGTAcc = float(sum(gtAcc))/len(gtAcc)
        accQA.append(avgGTAcc)
        if quesType not in accQuesType:
            accQuesType[quesType] = []
        accQuesType[quesType].append(avgGTAcc)
        if ansType not in accAnsType:
            accAnsType[ansType] = []
        accAnsType[ansType].append(avgGTAcc)

    accuracy = round(100*float(sum(accQA))/len(accQA), 2)
    accQuesType = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), 2) for quesType in accQuesType}
    accAnsType = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), 2) for ansType in accAnsType}
    
    return accuracy,accQuesType, accAnsType 


### call evaluate

qId = generated.keys()
accuracy,accQuesType, accAnsType = evaluate(qId, target, generated)
print(accuracy,accQuesType, accAnsType)

