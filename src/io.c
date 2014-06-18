#include <stdio.h>
#include <stdlib.h>
#include <csv.h>

#define BUF_SIZE (1024)

/*
** TODO
** Wrap up user data in structure. Signal ``abort'' to CSV reader.
*/
int ioReadCSV(
	const char * fileName,
	void (*cbField)(void *, size_t, void *),
	void (*cbRow)(int, void *),
	void * data
){
	// try to open file
	FILE * file = fopen(fileName, "r");
	if(!file){
		printf("Could not open CSV file\n");
		return -1;
	}

	// init csv parser
	int csvStatus;
	struct csv_parser csvParser;

	csvStatus = csv_init(&csvParser, CSV_APPEND_NULL);
	if(csvStatus){
		printf("Could not init csv parser\n");
		return -1;
	}

	// set up buffer
	size_t bufRead;
	size_t bufParsed;
	char buf[BUF_SIZE];
	
	while(1){
		bufRead = fread(buf, sizeof(char), BUF_SIZE, file);

		// reached end of file?
		if(!bufRead){
			break;
		}

		// try to parse buffer
		bufParsed = csv_parse(
			// pointer to parser
			&csvParser,
			// buffer and length
			buf, bufRead,
			// callback functions
			cbField,
			cbRow,
			// state
			data
		);

		if(bufParsed < bufRead){
			csvStatus = csv_error(&csvParser);
			printf("Error while parsing CSV file. Error:\n");
			printf("%s\n", csv_strerror(csvStatus));

			fclose(file);
			csv_free(&csvParser);
			return -1;
		}
	}

	fclose(file);
	csv_free(&csvParser);
	return 0;
}

struct matReadState {
	float * mat;
	int rows;
	int cols;
	int curRow;
	int curCol;
};

void ioReadMatFieldCB(void * buf, size_t bufLen, void * data){
	struct matReadState * statePtr = (struct matReadState *) data;
	if(statePtr->curCol >= statePtr->cols){
		printf("Matrix in CSV file has too many columns.\n");
		return;
	}

	if(statePtr->curRow >= statePtr->rows){
		printf("Matrix in CSV file has too many rows.\n");
		return;
	}

	float val;
	int read;

	read = sscanf((char *) buf, "%f", &val);
	if(!read){
		printf("Field in CSV file is not floating point number.\n");
		return;
	}

	/*
	** TODO
	** assume that matrix is stored in column major format
	*/
	size_t offset = 0;
	offset += statePtr->cols * statePtr->curCol;
	offset += statePtr->curRow;

	// store value
	statePtr->mat[offset] = val;

	// go to next field
	statePtr->curCol++;
}

void ioReadMatRowCB(int lineBreak, void * data){
	struct matReadState * statePtr = (struct matReadState *) data;

	statePtr->curCol = 0;
	statePtr->curRow++;
}

int ioReadMat(
	const char * fileName,
	float * mat,
	int rows,
	int cols
){
	int status;
	struct matReadState state = {
		.mat = mat,
		.rows = rows,
		.cols = cols,
		.curRow = 0,
		.curCol = 0
	};

	status = ioReadCSV(
		fileName,
		ioReadMatFieldCB,
		ioReadMatRowCB,
		(void *) &state		
	);

	// check for errors
	if(status){
		printf("Could not read matrix from CSV file.\n");
		return status;
	}

	return 0;
}

struct matSize {
	int rows;
	int cols;
};

void ioReadMatSizeColCB(void * buf, size_t bufLen, void * data){
	struct matSize * matSizePtr = (struct matSize *) data;

	// count columns of first row
	if(!matSizePtr->rows){
		matSizePtr->cols++;
	}
}

void ioReadMatSizeRowCB(int lineBreak, void * data){
	struct matSize * matSizePtr = (struct matSize *) data;

	// increment number of rows
	matSizePtr->rows++;
}

int ioReadMatSize(const char * fileName, int * rows, int * cols){
	int status;
	struct matSize matSize = {
		.rows = 0,
		.cols = 0
	};

	status = ioReadCSV(
		fileName,
		ioReadMatSizeColCB,
		ioReadMatSizeRowCB,
		(void *) &matSize
	);

	if(status){
		printf("Failed to read matrix size from CSV file.\n");
		return status;
	}

	*rows = matSize.rows;
	*cols = matSize.cols;

	return 0;
}
