from datasets import load_dataset
import pandas as pd

dataset = load_dataset("quora")

print(dataset.keys())
print("=======================================================================================")

raw_df = dataset["train"].to_pandas()

print(raw_df)
print("=======================================================================================")

raw_df = raw_df.loc[raw_df['is_duplicate']==True].reset_index(drop=True)

print(raw_df)
print("=======================================================================================")

# 중복되는 id를 개별 컬럼으로 배치
raw_df["q1"] = raw_df["questions"].apply(lambda q: q["text"][0])
raw_df["q2"] = raw_df["questions"].apply(lambda q: q["text"][1])
raw_df["id1"] = raw_df["questions"].apply(lambda q: q["id"][0])
raw_df["id2"] = raw_df["questions"].apply(lambda q: q["id"][1])

q1_to_q2 = raw_df.copy().rename(columns={"q1": "text", "id1": "id", "id2": "dq_id"}).drop(columns=["questions", "q2"])
q2_to_q1 = raw_df.copy().rename(columns={"q2": "text", "id2": "id", "id1": "dq_id"}).drop(columns=["questions", "q1"])
flat_df = pd.concat([q1_to_q2, q2_to_q1])

flat_df = flat_df.sort_values(by=['id']).reset_index(drop=True)

print(flat_df)
print("=======================================================================================")

print(flat_df.loc[((flat_df['id'] <= 15000) & (flat_df['dq_id'] <= 15000))])
print("=======================================================================================")

# 각 질문 하나당 중복되는 질문 id를 list 형태로 저장
df = flat_df.drop_duplicates("id")
df.loc[:, "duplicated_questions"] = df["id"].apply(lambda qid: flat_df[flat_df["id"] == qid]["dq_id"].tolist())
df = df.drop(columns=["dq_id", "is_duplicate"])
df.loc[:, 'length'] = [len(x) for x in df['duplicated_questions']]

print(df.loc[[len(i)>2 for i in df.duplicated_questions]])
print("=======================================================================================")

df.to_csv("datasets/quora_dataset.csv", index=False)