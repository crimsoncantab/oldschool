/*
 * Example for CS143 Programming Assignment One showing how to
 * read from the standard input as well as a socket.
 *
 * Note that while you will probably want to read the
 * standard input with the stdio package (possibly fgets), you
 * have to pass a file descriptor (not a FILE pointer such as stdin)
 * to select.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/*
 * Packet format shared by pa-one-send and pa-one-recv.
 */
struct packet {
  int field_one;
  int field_two;
};

main(int argc, char *argv[])
{
  fd_set mask;
  struct packet p;
  char buf[512];
  int n, id, port, s, len, cc;
  struct sockaddr_in sin;
  struct timeval tv;

  if(argc != 2 || (port = atoi(argv[1])) <= 0){
    fprintf(stderr, "Usage: pa-one-recv port\n");
    exit(1);
  }

  s = socket(AF_INET, SOCK_DGRAM, 0);
  if(s < 0){
    perror("pa-one-recv: socket");
    exit(1);
  }

  memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  if(bind(s, (struct sockaddr *)&sin, sizeof(sin)) < 0){
    perror("pa-one-recv: bind");
    exit(1);
  }

  while(1){
    FD_ZERO(&mask);
    FD_SET(fileno(stdin), &mask);
    FD_SET(s, &mask);
    tv.tv_sec = 1; /* time out after a second. */
    tv.tv_usec = 0;
    n = select(s+1, &mask, (fd_set*)0, (fd_set*)0, &tv);
    if(n < 0){
      perror("select");
      exit(1);
    }

    if(n == 0){
      printf("Timed out.\n");
    }

    if(FD_ISSET(fileno(stdin), &mask)){
      if(fgets(buf, sizeof(buf), stdin) == 0)
	exit(0); /* end of file */
      id = atoi(buf);
      printf("Read %d from stdin.\n", id);
    }

    if(FD_ISSET(s, &mask)){
      len = sizeof(sin);
      cc = recvfrom(s, &p, sizeof(p), 0,
                    (struct sockaddr *)&sin, &len);
      if(cc < 0){
        perror("pa-one-recv: recvfrom");
        exit(1);
      }
      printf("Received an %d byte packet from host %s port %d.\n",
             cc,
             inet_ntoa(sin.sin_addr),
             ntohs(sin.sin_port));
      if(cc == sizeof(p)){
        printf("  field_one %d field_two %d\n",
               ntohl(p.field_one),
               ntohl(p.field_two));
      } else {
        printf("  The length is wrong.\n");
      }
    }
  }
}
