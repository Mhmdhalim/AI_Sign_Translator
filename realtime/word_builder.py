import time


class WordBuilder:

    def __init__(self):
        self.word = ""

        self.last_prediction = None
        self.last_time = 0

        self.delay = 1.0


    def add_letter(self, letter):

        current_time = time.time()


        if letter != self.last_prediction:
            self.last_prediction = letter
            self.last_time = current_time
            return


        if current_time - self.last_time >= self.delay:

            self.word += letter

            self.last_prediction = None
            self.last_time = current_time



    def delete_last(self):

        self.word = self.word[:-1]



    def clear(self):

        self.word = ""



    def get_word(self):

        return self.word