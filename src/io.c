#include <stdio.h>
#include <stdlib.h>
#include <csv.h>

#define BUF_SIZE (1024)

void ioReadMat(){

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

int ioReadMatSize(char * fileName, int * rows, int * cols){
	// try to open file
	FILE * file = fopen(fileName, "r");
	if(!file){
		printf("Could not open file\n");
		return -1;
	}

	// init csv parser
	struct csv_parser csvParser;
	int csvStatus = 0;

	csvStatus = csv_init(&csvParser, CSV_APPEND_NULL);
	if(csvStatus){
		printf("Could not init csv parser\n");
		return -1;
	}

	size_t bufRead;
	char buf[BUF_SIZE];
	struct matSize matSize;
	
	while(1){
		bufRead = fread(buf, sizeof(char), BUF_SIZE, file);

		// reached end of file?
		if(!bufRead){
			break;
		}

		csvStatus = csv_parse(
			// pointer to parser
			&csvParser,
			// buffer and length
			buf, bufRead,
			// callback functions
			ioReadMatSizeColCB,
			ioReadMatSizeRowCB,
			// matrix size data
			(void *) &matSize
		);

		if(!csvStatus){
			printf("Error in CSV parser. Error:\n");
			printf("%s\n", csv_strerror(csv_error(&csvParser)));

			fclose(file);
			return -1;
		}
	}

	fclose(file);

	// write result
	*rows = matSize.rows;
	*cols = matSize.cols;

	return 0;
}
