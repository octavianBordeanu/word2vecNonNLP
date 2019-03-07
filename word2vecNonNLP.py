from lib import *


class Word2VecNonNLP(object):

    def __init__(self):

        self.window_size = 1
        self.learning_rate = 0.05
        self.epochs = 2
        self.embedding_size = 150
        self.batch_training = True
        self.batch_size = 32
        self.min_count = 5
        self.subsample_freq_words = True
        self.freq_threshold = 1e-3
        self.distortion = 0.75
        self.n_samples = 5
        self.model = 'cbow'
        self.optimiser = 'adagrad'
        self.embeddings_to_csv = False

        self.indexed_sentences = None
        self.centre_context_pairs = []
        self.index_to_word = dict()
        self.word_to_index = dict()
        self.vocabulary = None
        self.vocabulary_size = None
        self.negative_samples = None
        self.mapped_embeddings = None
        self.similarities = None

        self.input_weights = None
        self.output_weights = None
        self.gradients_squared = None
        self.adapted_learning_rates = None
        self.rate_change = None
        self.loss = None

    @staticmethod
    def open_file():
        root = tk.Tk()
        root.withdraw()  # Do not open the root window.

        file = askopenfilename(initialdir=os.getcwd(),
                               filetypes=[("csv files", "*.csv"), ("All files", "*.*")],
                               title="Select the file to process.")
        root.update()  # Update root window otherwise it will not close after selecting the file.
        # Check if a csv file was selected.
        try:
            os.path.isfile(file) and fnmatch.fnmatch(file, "*.csv")
            data_ = pd.read_csv(file, index_col=0)
            print("File loaded.")
            root.destroy()
            return data_
        except ValueError:
            print("No file or incorrect file was selected.")

    @staticmethod
    def subset_data(corpus, subset_variables):
        return corpus.reindex(columns=subset_variables).reset_index(drop=True)

    @staticmethod
    def select_output_directory():
        root = tk.Tk()
        root.withdraw()  # Do not open the root window.

        output_directory = askdirectory(title="Select output directory.")
        root.update()  # Update root window otherwise it will not close after selecting the file.
        return os.path.realpath(output_directory)

    def _preprocess_corpus(self, corpus):
        """ The following improve the learning of word embeddings:

            (1) remove rare words (i.e. min count < 5 which is also used in gensim package);
            (2) subsample frequent words to reduce the training set and counter the imbalance between rare and
            frequent words;
            (3) randomise window size (not implemented in this code since we want the full window to be considered
            when creating the training set);
            (4) use negative sampling to update a small proportion of the weights for each training input (i.e.
            useful range is between 5 to 20 for small datasets and 2 to 5 for large datasets).

        By default, rare words are removed if they do not occur at least 5 times in the training set.

        Each word is subsampled (discarded from the training set) with a probability P(w) = 1 - sqrt(t / f(w)). See
        function _subsample_frequent_words() for more details.

        A unigram distribution for negative sampling is the created using function _create_unigrams(). More details
        are provided within the function body.
        """

        def remove_indices(row):
            return [index for index in row if index != 'UNK']

        def index_sentence(sentence):
            return [self.word_to_index[word] for word in sentence]

        train_words = np.array(corpus).flatten().tolist()

        if self.min_count > 0:
            train_words = self._remove_low_count_words(train_words)

        if self.subsample_freq_words:
            train_words = self._substample_frequent_words(train_words)

        train_words = np.array(train_words).reshape(corpus.shape).tolist()
        train_words = [remove_indices(row) for row in train_words]

        sentences = [row for row in train_words if len(row) > 1]
        train_words = [word for sentence in sentences for word in sentence]

        self.vocabulary = list(collections.OrderedDict.fromkeys([word for word in train_words if word != 'UNK']))

        # Dictionaries that map words to unique indices and viceversa.
        self.index_to_word = collections.OrderedDict(enumerate(self.vocabulary))
        self.word_to_index = {word: index for index, word in self.index_to_word.items()}

        # The training set.
        self.indexed_sentences = [index_sentence(sentence) for sentence in sentences]

        self.vocabulary_size = len(self.vocabulary)

        print("Corpus processed.")

        self._create_unigrams()

    def _remove_low_count_words(self, train_words):
        word_count = collections.Counter(train_words)
        return [word if word_count[word] >= self.min_count else 'UNK' for word in train_words]

    def _substample_frequent_words(self, train_words):
        """ Subsample frequent words to counter the imbalance between rare and frequent words by discarding each word
        in the in the training set with a probability given by P(w) = 1 - sqrt(t / f(w)), where

           • w is given word from the training set;
           • t is is a chosen threshold which determines how much subsampling occurs. Recommended value is 1e-5
           although lower values are also used, usually between the range [1e-3, 1e-5]. The gensim package, written by
           Radim Rehurek, uses 1e-3. The smaller the sample threshold the less likely is that words are kept in
           the training set.
           • f(w) is the frequency of a given word in the training set.

        *** Sources:
        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of
        words and phrases and their compositionality." In Advances in neural information processing systems, pp.
        3111-3119. 2013.
        """

        # The training set that is subsampled is that with the discarded infrequent words.
        words_to_subsample = [word for word in train_words if word != 'UNK']
        words_to_subsample_count = collections.Counter(words_to_subsample)
        total_words = len(words_to_subsample)
        frequencies = {word: count / total_words for word, count in words_to_subsample_count.items()}
        p_subsample = {word: 1 - np.sqrt(self.freq_threshold / frequencies[word]) for word in words_to_subsample_count}
        print("Frequent words subsampled.")
        return [word if word != 'UNK' and random.random() < (1 - p_subsample[word]) else 'UNK' for word in train_words]

    def _create_unigrams(self):
        """ Build the cumulative distribution table from were negative words are randomly drawn.

        By using a unigram distribution, we ensure that the randomly selected negative samples are more likely to be
        frequent words. Oversampling frequent words improves performance because they carry more information than
        less frequent words.

        The equation used for selecting negative samples is P(wi) = (f(wi)**3/4)/sum(f(wj)**3/4), where

            • wi is word i;
            • P(wi) is probability that word i is selected as negative sample.
            • f(wi) is the frequency of word i in the entire corpus;
            • sum(f(wj)) is the sum of frequencies for all words in the corpus.
            • Mikolov et al.(2013) and Levy et al. (2015) found that raising the unigram distribution to the
            power of 3/4 performed better than the uniform and unigram distributions.

        *** Sources:
        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of
        words and phrases and their compositionality." In Advances in neural information processing systems, pp.
        3111-3119. 2013.

        Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving distributional similarity with lessons learned from word
        embeddings." Transactions of the Association for Computational Linguistics 3 (2015): 211-225.

        McCormick, C. (2017, January 11). Word2Vec Tutorial Part 2 - Negative Sampling.
        Retrieved from http://www.mccormickml.com
        """

        training_words_count = collections.Counter([index for pair in self.indexed_sentences for index in pair])
        training_words_pow = sum([count ** self.distortion for count in training_words_count.values()])

        # The number of times each word appears in the table is P(w) * table_size, that is int(1e8 * P(w)).
        # Then generate random number between 0 and 100 million, then use the word at that index in the table.
        word_counts = list(training_words_count.values())
        probs = np.zeros(self.vocabulary_size)
        for word_index, word_count in enumerate(word_counts):
            prob = (word_count ** self.distortion) / training_words_pow
            probs[word_index] = prob

        negative_samples = []
        for index, prob in enumerate(probs):
            negative_samples.extend([index] * int(prob * 1e8))

        self.negative_samples = np.array(negative_samples)
        print("Negative samples table created.")

    def _sample(self):
        # Pick a random number between 0 and size of negative samples table.
        negative_samples = np.random.randint(low=0, high=len(self.negative_samples), size=self.n_samples)
        return [self.negative_samples[i] for i in negative_samples]

    def _target_context_generator(self):
        """ Create (target, context) pairs. For example, given the following indexed sentence and a window size of 1,
        the (target, context) pairs are:

        Indexed sentence: [0, 1, 3, 6]

            • skipgram pairs: (0, 1), (1, 0), (1, 3), (3, 1), (3, 6), (6, 3) with the form (target, label).
            • cbow pairs: (0, [1]), (1, [0, 3]), (3, [1, 6]), (6, [3]) with the form (label, target).
        """
        self.centre_context_pairs = []
        for sentence in self.indexed_sentences:
            for i in range(len(sentence)):
                target = sentence[i]
                context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and len(sentence) - 1 >= j >= 0:
                        context.append(sentence[j])
                self.centre_context_pairs.append([target, context])

    def train(self, data):
        """ The training data is processed and the (target, context) pairs are generated before training the model.

        First, initialise the input and output weights matrices with random values between -1 and 1 drawn from a
        uniform distribution. Let V be vocabulary size, N the embedding size, x a word in V, and y the output label
        given input x. Then,

            • shape of input weights matrix W is V X N. This matrix gives the vector for input x.
            • shape of output weights W' is V X N. This matrix gives the vector for output (label) y. In academic
            papers and other tutorials on word2vec, the shape of the output weights matrix is given as N X V. This is
            because the mathematical algorithm uses one-hot vectors to get the corresponding rows (embeddings) for
            both the input and output words from the input and output weights matrices, respectively. However, here we
            use words indices to identify their corresponding rows in both weights matrices. For example, for word with
            index 0 (first in the vocabulary), its input and output embeddings are at row 0 in both weights matrices,
            which means that the shape of both matrices can be kept the same.

        Second, loop through each (centre, context) pair -- can be used interchangeably with (input, label) or
        (centre, target), etc. -- and

            (1) if model is CBOW:

                • get the embeddings for all context indices given input word x. For example, assume [0, 2] to be
                context indices for word x1, given a window size of 1. For each context index in [0, 2] we
                get their embeddings from the input weights matrix and then average them -- that is,
                (row 0 + row 2) / (2 * window size). This way we get an average embedding for x1. Let's call this
                vector h;
                • get negative samples. Let the negative samples set be
                [(2, 1), (585, 0), (655, 0), (908, 0), (760, 0), (106, 0)], where the first element in each pair is
                the sampled word index and second element each in pair denotes whether or not that word index is
                indeed a label for the given input (0 true and 1 false). Here we have 6 samples rather than 5 because
                the first is a positive sample and the other 5 are the negative samples.

                    - (2, 1) index 1 is a true label for the given input which is 2;
                    - (585, 0) index 585 is a false label for the given input which is 2.

                • calculate a score vector, let it be z, given h which is the dot product between the output weights
                matrix and h -- that is, z = W' • h. This gives the output vector representation for x1 given its
                context [0, 2] as input. This is also calculated for each negative sample.

                • the sigmoid function then generates a probability that the output word (associated with each
                output given the positive and negative inputs) is a true label for the input word. This probability
                is then multiplied by the learning rate to get the propagation error.

                • the backpropagation is then calculated by multiplying the difference between the current label and
                its probability and then multiply it by the learning rate, for both positive and negative samples.
                These errors are then used to update both input and output weight matrices (W, W').

            (2) if model is Skipgram:

                • follows the same approach as CBOW the difference being that here the input is the centre word and
                not the context word/words. In CBOW we attempt to get the centre word given its context words while
                in skipgram we attempt to get the context words given the centre word.
                • the hidden layer is therefore the dot product between each centre and context word in the pair
                (centre, context).

        *** Important: word2vec does not require a large number of epochs to produce quality embeddings:
            • Gensim uses 5 epochs by default;
            • the larger the corpus the less iterations are needed;
            •


        *** Sources:
        Chia, D. (2018, December 6). An implementation guide to Word2Vec using NumPy and Google Sheets. Retrieved from
        https://tinyurl.com/y5grqwdq

        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of
        words and phrases and their compositionality." In Advances in neural information processing systems, pp.
        3111-3119. 2013.

        Meyer, David. "How exactly does word2vec work?." (2016). Retrieved fronm
        http://www.1-4-5.net/~dmm/ml/how_does_word2vec_work.pdf

        Rong, Xin. "word2vec parameter learning explained." arXiv preprint arXiv:1411.2738 (2014).

        Sarkar, D.(2018, May 14). Understanding Feature Engineering (Part 4) — A hands-on intuitive approach to Deep
        Learning Methods for Text Data — Word2Vec, GloVe and FastText. Retrieved from https://tinyurl.com/yxoxvhva

        Socher, R. et al.(2016) CS 224D: Deep Learning for NLP. Lecture Notes: Part I. Retrieved from
        https://cs224d.stanford.edu/lecture_notes/notes1.pdf
        """

        self._preprocess_corpus(data)
        self._target_context_generator()

        self.input_weights = np.random.uniform(low=-1.0, high=1.0, size=(self.vocabulary_size, self.embedding_size))
        self.output_weights = np.zeros(shape=(self.vocabulary_size, self.embedding_size))

        self.gradients_squared = np.zeros(shape=(self.vocabulary_size,))
        self.adapted_learning_rates = np.full((self.vocabulary_size,), self.learning_rate)

        if self.batch_training:
            if len(self.centre_context_pairs) % self.batch_size == 0:
                iterations = len(self.centre_context_pairs) // self.batch_size
            else:
                iterations = (len(self.centre_context_pairs) // self.batch_size) + 1
        else:
            iterations = 1

        print("Training...")
        for epoch in range(self.epochs):
            self.loss = 0
            for iteration in range(iterations):
                if self.batch_training:
                    training_set = self._generate_batch(iteration, iterations)
                else:
                    training_set = self.centre_context_pairs

                for index, pair in enumerate(training_set):
                    centre = training_set[index][0]
                    context = training_set[index][1]

                    if self.model == 'cbow':
                        self._cbow_train(centre, context)
                    else:
                        self._sg_train(centre, context)

            print("Epoch {}, cumulative loss {}".format(epoch+1, self.loss))

    def _generate_batch(self, iteration, iterations):
        if iteration in range(iterations - 1):
            batch = self.centre_context_pairs[iteration * self.batch_size:iteration * self.batch_size + self.batch_size]
        else:
            batch = self.centre_context_pairs[iteration * self.batch_size:]

        return batch

    def _cbow_train(self, centre, context):
        negative_samples = [(centre, 1)] + [(target, 0) for target in self._sample()]
        # Average context vectors for given centre word.
        input_layer = np.mean(np.array([self.input_weights[c] for c in context]), axis=0)
        # Empty array to store the error gradients.
        input_errors = np.zeros(self.embedding_size)
        for target, label in negative_samples:
            # Get the vector representation of the input.
            input_score = np.dot(input_layer, self.output_weights[target])

            # Propagate hidden layer to output layer.
            # This turns the dot product into a probability which gives the probability that the
            # hidden layer (label vector) is a real label for the given context.
            prob = self._sigmoid(input_score)

            # Output word loss.
            if label == 1 and prob != 0:
                self.loss -= np.log(prob)

            # Gives the difference between the input and the output (error or loss) multiplied by the learning rate.
            # Also known as gradient.
            if self.optimiser == 'adagrad':
                # Adaptive learning rate.
                learning_rate = self._adjust_learning_rate(target, label - prob)
            else:
                # Constant learning rate.
                learning_rate = self.learning_rate

            propagation_error = (label - prob) * learning_rate

            # The errors to be backpropagated to the input weights.
            input_errors += propagation_error * self.output_weights[target]

            # Update output weights.
            self.output_weights[target] += propagation_error * input_layer

        # Update input weights.
        for context_word in context:
            self.input_weights[context_word] += input_errors

    def _sg_train(self, centre, context):
        for context_word in context:
            input_errors = np.zeros(self.embedding_size)
            negative_samples = [(centre, 1)] + [(target, 0) for target in self._sample()]
            for target, label in negative_samples:
                input_score = np.dot(self.input_weights[context_word], self.output_weights[target])
                prob = self._sigmoid(input_score)

                if label == 1 and prob != 0:
                    self.loss -= np.log(prob)

                if self.optimiser == 'adagrad':
                    learning_rate = self._adjust_learning_rate(label - prob, target)
                else:
                    learning_rate = self.learning_rate

                propagation_error = (label - prob) * learning_rate
                input_errors += propagation_error * self.output_weights[target]
                self.output_weights[target] += propagation_error * self.input_weights[context_word]
            self.input_weights[context_word] += input_errors

    @staticmethod
    def _sigmoid(x):
        """ The sigmoid function takes a real number and outputs it in a range between 0 and 1, and is used in the
        negative sampling objective function (see equation 4 on page 3 in Mikolov et al. 2013).

            • The output for large negative numbers (x < -6) becomes 0 (approximately 0.0025).
            • The output for large positive numbers (x > 6) becomes 1 (approximately 0.9975).
            • The output for numbers between -6 and 6 (-6 <= x <= 6) becomes 1 / (1 + e^-x).

        *** Sources:
        https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
        https://isaacchanghau.github.io/post/word2vec/

        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of
        words and phrases and their compositionality." In Advances in neural information processing systems, pp.
        3111-3119. 2013.
        """
        if x > 6:
            return 1.0
        elif x < -6:
            return 0.0
        else:
            return 1 / (1 + math.exp(-x))

    def _adjust_learning_rate(self, target, propagation_error):
        """ AdaGram optimisation algorithm, proposed by Duchi et al. (2011), has proved to perform better than keeping
        the learning rate constant or using linear approaches for optimising the learning rate, such as stochastic
        gradient descent (Lau (2017), Ward et al.(2018)). It ensures that learning rate is neither too small nor too
        large which could make the model take longer to converge to a local/global minima, or even get stuck in a local
        minima. This problem is known as the vanishing and exploding gradient problem, which AdaGram can handle well.

        AdaGram is an adaptive optimisation algorithm which adapts the learning rate for each input separately. For
        each input, the sum of squared gradients is saved which is used to scale the learning rate for that input. The
        formula used to scale the learning rate for each training input is:

                alpha_i = alpha / sqrt(G_i + epsilon), where

        • alpha_i is the adjusted learning rate for input i;
        • alpha is previous a constant learning rate set before training;
        • G_i is the sum of all squared gradients for input i up to date;
        • epsilon is a very small number (usually 1e-8)  that is used to avoid division by zero in case G_i is zero.

            * the formula used here was from page 2 in Schaul et al. (2013) because it was presented in a more friendly
            and easy to understand manner than other papers that are highly mathematical, thus easier to understand how
            implement to implement it.

        AdaGrad ensures that larger learning rates updates are performed for infrequent inputs and smaller learning
        rate updates are performed for frequent inputs.

        *** Sources
        Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online learning and stochastic
        optimization." Journal of Machine Learning Research 12, no. Jul (2011): 2121-2159.

        Lau, S. (2017, July 29). Learning Rate Schedules and Adaptive Learning Rate Methods for Deep Learning.
        Retrieved from https://tinyurl.com/y7ezycfx

        Schaul, Tom, Sixin Zhang, and Yann LeCun. "No more pesky learning rates." In International Conference on
        Machine Learning, pp. 343-351. 2013.

        Ward, Rachel, Xiaoxia Wu, and Leon Bottou. "Adagrad stepsizes: Sharp convergence over nonconvex landscapes,
        from any initialization." arXiv preprint arXiv:1806.01811 (2018).
        """
        # Update gradients history for input word index.
        self.gradients_squared[target] += propagation_error ** 2
        # Adjust learning rate for given input.
        self.adapted_learning_rates[target] = self.learning_rate / (self.gradients_squared[target] + 1e-8)

        return self.adapted_learning_rates[target]

    def similar(self, query_word, to_csv):

        def cosine_distance(query_vector, vector):
            return np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))

        similarities = [cosine_distance(self.output_weights[self.word_to_index[query_word]], vector) for
                        vector in self.output_weights]

        output = pd.DataFrame({'word': [word for word in self.vocabulary],
                               'index': [index for index in self.index_to_word.keys()],
                               'similarity': similarities}).sort_values(['similarity'], ascending=False)

        if to_csv:
            output_dir = self.select_output_directory()
            print("Select the directory where the output is to be saved.")
            output.drop(output.index[0]).to_csv(f"{output_dir}/{query_word}_similarities.csv", sep=",")

        # Drop first row because it contains the similarity between the query vector and itself which is 1.
        return output.drop(output.index[0])

    def predict_word(self, words, to_csv):
        """ This function uses softmax to get a probability distribution for each centre word (word in vocabulary)
        given the context words (inputted words in the *words parameter of this function). In other words, the
        output layer is a softmax layer which assigns to each class (centre word) a probability that it is the
        centre word for the given context.

        Process: (1) Calculate average vector for the context words.
                     h = (v1 + v2+ ... + vn) / n

                 * Steps 2, 3, 4 are for the softmax calculation.

                 (2) Calculate the the score for the average vector of context words and all other words in vocabulary.
                 This is given by the dot product between h and the output weights matrix which contains the output
                 embeddings for each word in vocabulary.
                     z = W'(dot)h

                 (2) Calculate the exponential values of the averaged vector representing the context words;
                     exp(z)

                 (3) Calculate the sum of exponential values of z.
                     sum(exp(z))

                 (4) Divide (2) by (3) to get the probabilities that the averaged vector for the given context
                 corresponds to each centre word (word in vocabulary).

                     probabilities = exp(z) / sum(exp(z))
        """
        # Compute average of given context vectors.
        context_vectors_avg = np.mean(np.array([self.output_weights[self.word_to_index[word]] for word in words]),
                                      axis=0)

        # Exclude words used for prediction.
        indices = [v for k, v in self.word_to_index.items() if k not in words]

        # Get a score for the context averaged vector with respect to the vector of all other words in vocabulary.
        context_scores = np.dot(context_vectors_avg, (self.output_weights[indices, :]).T)

        # Turn scores into probabilities using softmax.
        exp_context_scores = np.exp(context_scores)
        probabilities = exp_context_scores / sum(exp_context_scores)

        output = pd.DataFrame({'word': [word for word in self.vocabulary if word not in words],
                               'index': [index for index in indices],
                               'probability': probabilities}).sort_values(['probability'], ascending=False)

        if to_csv:
            output_dir = self.select_output_directory()
            print("Select the directory where the output is to be saved.")
            output.drop(output.index[0]).to_csv(f"{output_dir}/probabilities.csv", sep=",")

        return output

    @staticmethod
    def _reduce_vector_dimensions(word_vectors, n_dim):
        vectors = word_vectors.loc[:, 'dimension_1':]
        pca = PCA(n_components=n_dim, random_state=1)
        scaled_data = StandardScaler().fit_transform(vectors)
        pca_dim = pca.fit_transform(scaled_data)

        return pca_dim

    def embeddings(self, reduce_dim, n_dim, to_csv):
        dimensions = [f'dimension_{i + 1}' for i in range(self.embedding_size)]
        vectors_df = pd.DataFrame(self.output_weights, columns=dimensions)

        output = pd.DataFrame({'word': [word for word in self.vocabulary],
                               'index': [index for index in self.index_to_word.keys()]})
        output = pd.concat([output, vectors_df], axis=1, sort=False)

        if reduce_dim:
            reduced_dimensions = self._reduce_vector_dimensions(vectors_df, n_dim)
            reduced_dimensions = pd.DataFrame(reduced_dimensions,
                                              columns=[f'pca_dimension_{i + 1}' for i in range(n_dim)])
            output = pd.concat([reduced_dimensions, output], axis=1)

        if to_csv:
            output_dir = self.select_output_directory()
            print("Select the directory where the output is to be saved.")
            self.mapped_embeddings.to_csv(f"{output_dir}/word_vectors.csv", sep=",")

        return output


