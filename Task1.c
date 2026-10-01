/*
============================================================
Author: Oladipupo Roland Familua
Course: BSc Computer Science
Module: Operating Systems / Parallel Programming

Description:
This program uses POSIX threads (Pthreads) to count the
occurrence of words stored in a text file. The file name and
number of threads are provided via the command line.

The workload is divided across multiple threads, with each
thread processing a distinct portion of the input data.
A mutex is used to ensure thread-safe updates to shared data.
============================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

/* -----------------------------------------------------------
   STRUCTURES
----------------------------------------------------------- */

/*
ThreadData:
Stores information passed to each thread, including:
- The list of words/lines to process
- The start and end indices for this thread's workload
- A mutex for synchronising access to shared data
*/
typedef struct {
    char ** wordsList;
    int start, end;
    pthread_mutex_t *mutex;
} ThreadData;

/*
WordCount:
Stores a word and the number of times it appears in the file
*/
typedef struct {
    char word[32];
    long count;
} WordCount;

/* -----------------------------------------------------------
   GLOBAL VARIABLES
----------------------------------------------------------- */

/*
allWords:
Global array storing unique words and their frequencies.
This array is shared by all threads.
*/
WordCount * allWords = NULL;

/*
allWordsCount:
Tracks the number of unique words stored in allWords.
*/
int allWordsCount = 0;

/* -----------------------------------------------------------
   THREAD FUNCTION
----------------------------------------------------------- */

/*
processLines:
This function is executed by each thread.
Each thread processes a specific range of indices in the
wordsList array and updates the global word count.

A mutex is used to ensure that only one thread at a time
can modify the shared allWords array.
*/
void *processLines(void * arg) {
    ThreadData * data = (ThreadData *) arg;

    /* Iterate through the assigned portion of the word list */
    for (int i = data->start; i <= data->end; i++) {

        /* Get the current word/line */
        char * word = data->wordsList[i];

        /* Lock mutex before accessing shared data */
        pthread_mutex_lock(data->mutex);

        int found = 0;

        /* Search for the word in the global word list */
        for (int i = 0; i < allWordsCount; i++) {
            if (strcmp(allWords[i].word, word) == 0) {
                allWords[i].count++;
                found = 1;
                break;
            }
        }

        /* If the word is not found, add it to the list */
        if (found == 0) {
            strcpy(allWords[allWordsCount].word, word);
            allWords[allWordsCount].count = 1;
            allWordsCount++;
        }

        /* Unlock mutex after updating shared data */
        pthread_mutex_unlock(data->mutex);
    }

    /* Terminate thread execution */
    pthread_exit(NULL);
}

/* -----------------------------------------------------------
   MAIN FUNCTION
----------------------------------------------------------- */

int main (int argc, char * argv[]) {

    /* Validate command-line arguments */
    if (argc != 3) {
        printf("Usage: %s <filename> <num_threads>\n", argv[0]);
        return 1;
    }

    /* Open the input file */
    FILE * file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Unable to open provided file\n");
        exit(0);
    }

    /* -------------------------------------------------------
       FILE READING
    ------------------------------------------------------- */

    /*
    Count how many lines are in the file.
    Each line is treated as a single word entry.
    */
    int numberOfLines = 0;
    char ** wordsList;
    char temp[32];

    while (fgets(temp, sizeof(temp), file))
        numberOfLines++;

    /* Allocate memory to store each line */
    wordsList = malloc(sizeof(char *) * numberOfLines);

    /* Reset file pointer to the beginning */
    rewind(file);

    /*
    Read each line into the wordsList array
    and remove the newline character.
    */
    for (int i = 0; i < numberOfLines; i++) {
        wordsList[i] = malloc(32);
        fgets(wordsList[i], 32, file);
        wordsList[i][strcspn(wordsList[i], "\n")] = '\0';
    }

    /* -------------------------------------------------------
       THREAD CREATION AND WORK DISTRIBUTION
    ------------------------------------------------------- */

    /* Get number of threads from command line */
    int numberOfThreads = atoi(argv[2]);

    pthread_t threads[numberOfThreads];
    ThreadData thread_data[numberOfThreads];

    /*
    Calculate how many lines each thread should process.
    The remainder is distributed among the first few threads.
    */
    const int chunkSize = numberOfLines / numberOfThreads;
    const int remainder = numberOfLines % numberOfThreads;

    /* Initialise mutex for shared data access */
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    /* Allocate memory for global word storage */
    allWords = malloc(sizeof(WordCount) * numberOfLines);

    int start = 0;

    /*
    Create threads and assign each one a range of data
    */
    for (int i = 0; i < numberOfThreads; i++) {
        thread_data[i].start = start;
        thread_data[i].end = start + chunkSize - 1 + (i < remainder ? 1 : 0);
        thread_data[i].mutex = &mutex;
        thread_data[i].wordsList = wordsList;

        start = thread_data[i].end + 1;

        pthread_create(&threads[i], NULL, processLines, &thread_data[i]);
    }

    /* Wait for all threads to finish */
    for (int i = 0; i < numberOfThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* -------------------------------------------------------
       OUTPUT RESULTS
    ------------------------------------------------------- */

    /* Write final word counts to output file */
    FILE * outputFile = fopen("result.txt", "w");
    if (outputFile == NULL) {
        printf("Unable to open output file\n");
        exit(0);
    }

    for (int i = 0; i < allWordsCount; i++) {
        fprintf(outputFile, "%s: %ld\n",
                allWords[i].word,
                allWords[i].count);
    }

    /* -------------------------------------------------------
       CLEANUP
    ------------------------------------------------------- */

    fclose(file);
    fclose(outputFile);

    pthread_mutex_destroy(&mutex);

    for (int i = 0; i < numberOfLines; i++)
        free(wordsList[i]);

    free(wordsList);
    free(allWords);

    return 0;
}
