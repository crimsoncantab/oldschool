/*
 * pa-one-send host port
 *
 * Sample UDP sender for CS143 Programming Assignment One.
 * Sends a packet in a format that pa-one-recv expects.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

/*
 * Packet format shared by pa-one-send and pa-one-recv.
 */
struct packet {
  int field_one;
  int field_two;
};

main(int argc, char *argv[])
{
  struct hostent *hp;
  struct sockaddr_in sin;
  int s;
  struct packet p;

  if(argc != 3){
    fprintf(stderr, "Usage: pa-one-send host port\n");
    exit(1);
  }

  memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(atoi(argv[2]));

  hp = gethostbyname(argv[1]);
  if(hp == 0){
    fprintf(stderr, "pa-one-send: no host %s\n", argv[1]);
    exit(1);
  }
  memcpy(&sin.sin_addr, hp->h_addr, sizeof(sin.sin_addr));

  s = socket(AF_INET, SOCK_DGRAM, 0);
  if(s < 0){
    perror("pa-one-send: socket");
    exit(1);
  }

  p.field_one = htonl(1111);
  p.field_two = htonl(2222);

  if(sendto(s, &p, sizeof(p), 0, (struct sockaddr *)&sin, sizeof(sin)) < 0){
    perror("pa-one-send: sendto");
    exit(1);
  }

  exit(0);
}
