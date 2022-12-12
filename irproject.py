
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import pandas as pd
import numpy as np
import math
import warnings



warnings.filterwarnings("ignore")


stop_wards = stopwords.words('english')
stop_wards.remove('in')
stop_wards.remove('to')
stop_wards.remove('where')

# part_1
files_name = natsorted(os.listdir('DocumentCollection')) #read files name and save in list

doc_terms = []
for files in files_name:

    with open(f'DocumentCollection\{files}', 'r') as f:
        document = f.read()
    # print(document)
    tokenized_docs = word_tokenize(document)
    terms = []
    for word in tokenized_docs:
        if word not in stop_wards:
            terms.append(word)
    doc_terms.append(terms)
print("                 ***********************_word tokenized_***********************")
print(doc_terms)
print("*+" * 60)
print('\n')

# part_2
document_number = 0
positional_index = {} #create dictionary

for document in doc_terms:

    # For position and term in the tokens.
    for positional, term in enumerate(document):
        # print(pos, '-->' ,term)

        # If term already exists in the positional index dictionary.
        if term in positional_index:

            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1

            # Check if the term has existed in that DocID before.
            if document_number in positional_index[term][1]:
                positional_index[term][1][document_number].append(positional)

            else:
                positional_index[term][1][document_number] = [positional]

        # If term does not exist in the positional index dictionary
        # (first encounter).
        else:

            # Initialize the list.
            positional_index[term] = []
            # The total frequency is 1.
            positional_index[term].append(1)
            # The postings list is initially empty.
            positional_index[term].append({})
            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

    # Increment the file no. counter for document ID mapping
    document_number += 1
print('                ********************************_positional_index_*******************************')
print(positional_index)
print('*+' * 60)
print("\n")


'''################################# Query phrase  
query =input("Enter term  or query to Search: ")
final_list = [[] for i in range(10)]

for word in query.split():
    for key in positional_index[word][1].keys():

        if final_list[key-1] != []:
            if final_list[key-1][-1] ==positional_index[word][1][key][0]-1:
                final_list[key-1].append(positional_index[word][1][key][0])

        else:
            final_list[key-1].append(positional_index[word][1][key][0])

for position, list in enumerate(final_list,start=1):
    #print(position,list)
    if len(list)== len(query.split()):
        print(position)

'''




# part_3

all_word = []
for doc in doc_terms:
    for word in doc:
        all_word.append(word)

################Table 1
def get_term_freq(doc):
    words_found = dict.fromkeys(all_word, 0)
    for word in doc:
        words_found[word] += 1
    return words_found


term_freq = pd.DataFrame(get_term_freq(doc_terms[0]).values(), index=get_term_freq(doc_terms[0]).keys())

for i in range(1, len(doc_terms)):
    term_freq[i] = get_term_freq(doc_terms[i]).values()

term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print("                       *************_Term Frequency_*************")
print(term_freq)
print("\n")

############################Table2
def get_waighted_term_freq(x):
    if x > 0:
        return math.log(x) + 1
    return 0


for i in range(1, len(doc_terms) + 1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_waighted_term_freq)
print("               ****************Waighted_Term_Freq********************")
print(term_freq)

################## Table3 df\idf
tfd = pd.DataFrame(columns=['df', 'idf'])

for i in range(len(term_freq)):
    frequency = int(term_freq.iloc[i].values.sum())

    tfd.loc[i, 'df'] = frequency

    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

tfd.index = term_freq.index
print("\n\n")
print("             ******************df Idf******************")
print(tfd)


################## Table4
term_freq_inverse_doc_freq = term_freq.multiply(tfd['idf'], axis=0)

print("             ******************Tf*Idf******************")
print(term_freq_inverse_doc_freq)


########################### Table 5
document_length = pd.DataFrame()


def get_docs_length(col):
    return float(format(np.sqrt(term_freq_inverse_doc_freq[col].apply(lambda x: x ** 2).sum()), '.3f'))


for column in term_freq_inverse_doc_freq.columns:
    document_length.loc[0, column] = get_docs_length(column)  # '''+'len' '''

print("           *****************document_length******************")
print(document_length)


############################## Table6
normalized_term_freq_idf = pd.DataFrame()


def get_normalized(col, x):
    try:
        return x / document_length[col ].values[0]
    #   return format(x / document_length[col ].values[0],'.3f')
    except:
        return 0


for column in term_freq_inverse_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inverse_doc_freq[column].apply(lambda x : get_normalized(column, x))

print("\n\n")
print("               *********************Normalized tf.idf *******************************")
print(normalized_term_freq_idf)
print("\n\n")



def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except :
        return 0


option = int(input("Enter Option : 1 => Phrase Query or 0=> Exit : "))

while option != 0:
    if option == 1:
        q = input("Enter term  or query to Search: ").lower().strip().replace("_", " ")


        try:

            query = pd.DataFrame(index=normalized_term_freq_idf.index)
            query['tf'] = [1 if x in q.split() else 0 for x in list(normalized_term_freq_idf.index)]
            query['w_tf'] = query['tf'].apply(lambda x : get_w_tf(x))
            product = normalized_term_freq_idf.multiply(query['w_tf'],axis=0)
            query['idf'] = tfd['idf'] * query['w_tf']
            query['tf_idf'] = query['w_tf'] * query['idf']
            query['norm'] = 0

            for i in range(len(query)):
                query['norm'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))
            product2 = product.multiply(query['norm'], axis=0)


            math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]]))
            product2.loc[q.split()].values
            print(query)
            print("\n\n")

            query_len = np.sqrt(sum(query['idf'].values ** 2)) #calc q lenght
            print("Doc_len : ",end="   ")
            print(query_len)
            print("\n")

            scores = {}
            for col in product2.columns:
                if 0 in product2[col].loc[q.split()].values:
                    pass
                else:
                    scores[col] = product2[col].sum()
            print(scores )
            print("\n")

            prod_res = product2[list(scores.keys())].loc[q.split()]
            print(prod_res)
            print("\n")



            print("Cosin similarity")
            print(prod_res.sum())

            final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            print("\n")
            print("return docs : " ,end="  ")
            for doc in final_score:
                print(doc[0], end=' ')
            print("\n")
        except:
            print("Sorry,Not Found")

    option = int(input("Enter Option : 1 => Phrase Query or 0=> Exit : "))
else:
    print("Thanks For You")






