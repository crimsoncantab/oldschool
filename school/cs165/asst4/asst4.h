/*-
 * See the file LICENSE for redistribution information.
 *
 * Copyright (c) 2002,2007 Oracle.  All rights reserved.
 *
 * $Id: ex_apprec.h,v 12.5 2007/05/17 15:15:13 bostic Exp $
 */

#ifndef _ASST_4_
#define	_ASST_4_

//#include <db.h>
#include "asst4_auto.h"

/* Place function prototypes for log, read, print, recover functions. */
#define STRSIZE 16
#define DB_FILE "mydb.db"

typedef struct {
	char exists;
	char str[STRSIZE];
	int val;
	DB_LSN lsn;
	//char pad[3]; //not needed.  pads automatically for alignment
} record;
#define OFFSET(id) ((id-1) * sizeof(record))

int open_db();
void close_db(int);
void my_write(int __fd, const void * __buf, size_t __n);
void my_read(int __fd, void * __buf, size_t __nbytes);
void my_sync(int __fd);
int my_open(char *__file, int __oflag);
int my_open_s(char *__file, int __oflag, int __sflag);
void my_close(int __fd);
off_t my_seek(int __fd, __off_t __offset, int __whence);

#endif /* !_ASST_4_ */
