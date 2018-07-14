from tkinter import *
from tkinter.ttk import *
from help_functions import *
from random import randint
from nltk.tokenize.moses import MosesDetokenizer
from nltk import word_tokenize
import torch


detokenizer = MosesDetokenizer()


_, _, _, _, spans, text_para, text_ques, passages, questions, p_length, q_length = load_data('dev')

'''
	passages -> padded passages in the form of list of lists of numbers
	p_length -> length of unpaded passages
	q_length -> length of unpaded questions
	questions -> padded questions in the form of list of lists of numbers
	spans -> (start, end) tuples
	text_para -> list of lists of words in the passages  : text form
	text_ques -> list of lists of words in the question : text form

'''

Model = load_model()   ## SPECIFY THE PATH TO THE SAVED MODEL.
word_to_idx, _= word_idx_map()

def generate_random(*args):

	eg_num = randint(0, len(passages)-1)

	text1.delete('1.0', END)
	text2.delete('1.0', END)
	text1.insert(INSERT,str(detokenizer.detokenize(text_para[eg_num], return_str=True)))
	text2.insert(INSERT,str(detokenizer.detokenize(text_ques[eg_num], return_str=True)))

	eg_num1.set(eg_num)


def find_answer(*args):
	
	eg_number = (eg_num1.get())
	if eg_number != 0:

		batch_p_lengths = [p_length[eg_number]]
		batch_q_lengths = [q_length[eg_number]]

		max_val_pass_len = max([p_length[eg_number]])
		max_val_ques_len = max([q_length[eg_number]])

		batch_passages = [passages[eg_number][:max_val_pass_len]]
		batch_questions = [questions[eg_number][:max_val_ques_len]]

		val_text_para = text_para[eg_number]

	else:
		val_text_para = word_tokenize(text1.get('1.0', END))
		val_text_ques = word_tokenize(text2.get('1.0', END))
		
		batch_passages = [text_list_to_num_list(val_text_para, word_to_idx)]
		batch_questions = [text_list_to_num_list(val_text_ques, word_to_idx)]

		batch_p_lengths = [len(val_text_para)]
		batch_q_lengths = [len(val_text_ques)]

		max_val_pass_len = batch_p_lengths[0]
		max_val_ques_len = batch_q_lengths[0]


	start_outputs, end_outputs, _ = Model(max_val_pass_len, max_val_ques_len, batch_passages, batch_questions, batch_p_lengths,\
	batch_q_lengths, 1, False, False)

	if start_outputs.item() <= end_outputs.item():
		answer_str = detokenizer.detokenize(val_text_para[start_outputs.item():end_outputs.item() + 1], return_str=True)
		
	else:
		answer_str = detokenizer.detokenize(val_text_para[end_outputs.item():start_outputs.item() + 1], return_str=True)

	text3.delete('1.0', END)
	text3.insert(INSERT,answer_str)
		

root=Tk()
root.title("Question Answering System")
root.geometry("650x650")

passage_frame = Frame(root)
passage_frame.place(x=100, y=50)

question_frame = Frame(root)
question_frame.place(x=100, y=250)

answer_frame = Frame(root)
answer_frame.place(x=100, y=500)

passage = StringVar()
question = StringVar()
answer = StringVar()
eg_num1 = IntVar()

passage_heading = StringVar()
question_heading= StringVar()
answer_heading = StringVar()

text1 = Text(passage_frame, height= 10, width=60)
passage_label = Message(passage_frame, textvariable=passage_heading, width=100, relief = RAISED)
passage_heading.set("Passage")
passage_label.pack()
text1.pack()

text2 = Text(question_frame, height= 10, width=60)
question_label = Message(question_frame, textvariable=question_heading, width=100, relief = RAISED)
question_heading.set("Question")
question_label.pack()
text2.pack()

text3 = Text(answer_frame, height= 4, width=60)
answer_label = Message(answer_frame, textvariable=answer_heading, width=100, relief = RAISED)
answer_heading.set("Answer")
answer_label.pack()
text3.pack()


button1 = Button(root, text="Generate Random QandP", command=generate_random)
button1.place(x=100,y=450)

button2 = Button(root, text="Find Answer", command=find_answer)
button2.place(x=425,y=450)

root.mainloop()

