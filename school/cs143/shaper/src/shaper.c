#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <time.h>

#define TIMEOUT_SEC 0
#define TIMEOUT_USEC 100000
#define CLOCK_WAIT (CLOCKS_PER_SEC / 1000000 * TIMEOUT_USEC)
#define MAX_PACKET_SIZE 1600
//the number of packets a flow can hold onto at the same time
#define MAX_PACKETS 1000
//conversion ratio from Mbps to B/timeout
#define MBPS_TO_BYTE_P_T (125000.0 * TIMEOUT_USEC / 1000000.0)
#define MAX_NUM_FLOWS 10
#define TOKEN_CAPACITY 100000
//host where everything runs
#define HOST "localhost"

//typedefs for convenience
typedef struct sockaddr_in addr;
typedef in_port_t port;

typedef struct packet_buffer_ {
    char data[MAX_PACKET_SIZE];
    int size;
} packet_buffer;

typedef struct flow_ {
    addr in;
    addr out;
    int in_socket;
    int out_socket;
    int tokens;
    int tokens_per_ms;
    packet_buffer * packets[MAX_PACKETS];
    int next_buffer;
    int first_buffer;
} flow;

fd_set input_mask;
int max_socket;
flow * flows;
int num_flows;
struct hostent *hp;

//opens a socket, with error checking

int get_soc() {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    return s;
}


//send packet, "empty" buffer

void send_packet(packet_buffer * p, int socket, addr * address) {
    if (sendto(socket, p->data, p->size, 0,
            (struct sockaddr *) address, sizeof (addr)) < 0) {
        perror("sendto");
        exit(EXIT_FAILURE);
    }
    p->size = 0;
}

//fill up buffer with incoming packet

void recieve_packet(packet_buffer * p, int socket, addr * address) {
    int len = sizeof (addr);
    p->size = recvfrom(socket, p->data, sizeof (p->data), 0,
            (struct sockaddr *) address, &len);
    if (p->size < 0) {
        perror("recvfrom");
        exit(EXIT_FAILURE);
    }
}

//while parsing a string, go to index after colon

char * skip_next_colon(char * str) {
    int i = 0;
    while (str[i] != ':') i++;
    return &(str[++i]);
}
//parse args. for simplicity, DOES NOT CHECK INVALID ARGS

void init_flow(flow * f, char * args) {

    f->first_buffer = 0;
    f->next_buffer = 0;
    f->tokens = 0;

    int i;
    for (i = 0; i < MAX_PACKETS; i++) {
        packet_buffer * ptr = malloc(sizeof (packet_buffer));
        if (!ptr) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }
        f->packets[i] = ptr;
        f->packets[i]->size = 0;
    }

    f->in.sin_family = AF_INET;
    memcpy(&(f->in.sin_addr), hp->h_addr, sizeof (f->in.sin_addr));
    f->in.sin_port = htons(atoi(args));
    f->in_socket = get_soc();
    FD_SET(f->in_socket, &input_mask);
    max_socket = (max_socket > f->in_socket) ? max_socket : f->in_socket;

    if (bind(f->in_socket, (struct sockaddr *) & f->in, sizeof (f->in)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    args = skip_next_colon(args);

    float rate = atof(args);
    f->tokens_per_ms = rate * MBPS_TO_BYTE_P_T;
/*
        printf("%d\n", f->tokens_per_ms);
*/

    args = skip_next_colon(args);

    f->out.sin_family = AF_INET;
    memcpy(&(f->out.sin_addr), hp->h_addr, sizeof (f->out.sin_addr));
    f->out.sin_port = htons(atoi(args));
    f->out_socket = get_soc();

}

void parse_args(char**flow_args) {
    //TODO
    int i;
    for (i = 0; i < num_flows; i++) {
        init_flow(&(flows[i]), flow_args[i]);
    }

}

//initialize shaper state

void init(int argc, char**argv) {
    num_flows = (argc - 1 > MAX_NUM_FLOWS) ? MAX_NUM_FLOWS : argc - 1;
    flows = malloc(num_flows * sizeof (flow));
    if (!flows) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    max_socket = 0;
    FD_ZERO(&input_mask);

    hp = gethostbyname(HOST);
    if (hp == 0) {
        fprintf(stderr, "no host %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    parse_args(&argv[1]);
    //TODO
}


//called when router times out

void handle_timeout() {
    int i;
    for (i = 0; i < num_flows; i++) {
        flow * f = &(flows[i]);
        int new_tokens = f->tokens + f->tokens_per_ms;
        while (1) {
            int size = f->packets[f->first_buffer]->size;
            if (size == 0) break;
            if (size <= new_tokens) {
                new_tokens -= size;
                send_packet(f->packets[f->first_buffer++], f->out_socket, &(f->out));
                f->first_buffer %= MAX_PACKETS;
            } else {
                break;
            }
        }
        f->tokens = (new_tokens > TOKEN_CAPACITY) ? TOKEN_CAPACITY : new_tokens;

    }
}

//called when recieving data from stdin

void handle_stdin() {
    char buf[2];
    if (fgets(buf, sizeof (buf), stdin) == 0) {
        //eof
        exit(EXIT_SUCCESS);
    }
}

//called when recieving packet on listing port

void flow_recieve_packet(flow * f) {
    if (f->packets[f->next_buffer]->size == 0) {
        recieve_packet(f->packets[f->next_buffer++], f->in_socket, &f->in);
        f->next_buffer %= MAX_PACKETS;
/*
            } else {
                printf("full %d %d, %d\n", f->first_buffer, f->next_buffer, f->tokens);
*/
    }
}

void check_ports(fd_set * mask) {
    int i;
    for (i = 0; i < num_flows; i++) {
        if (FD_ISSET(flows[i].in_socket, mask)) {
            flow_recieve_packet(&(flows[i]));
        }
    }
}

//waits for activity on stdin or port

int wait(fd_set * mask, struct timeval * timer) {
    int n = select(max_socket + 1, mask, 0, 0, timer);
    /*
        printf("n %d\n", n);
     */
    if (n < 0) {
        perror("select");
        exit(EXIT_FAILURE);
    }
    return n;
}

//router looping function

void loop() {
    fd_set mask;
    struct timeval timer;
    timer.tv_sec = TIMEOUT_SEC;
    timer.tv_usec = TIMEOUT_USEC;
    clock_t time = clock();
/*
    printf("%d\n", CLOCKS_PER_SEC);
    printf("%d\n", CLOCK_WAIT);
*/
    while (1) {
        mask = input_mask;

        if (wait(&mask, &timer) == 0) {
            /*
                        handle_timeout();
             */
            timer.tv_sec = TIMEOUT_SEC;
            timer.tv_usec = TIMEOUT_USEC;
        }
        clock_t time2 = clock();
        if ((time2 - time) >= CLOCK_WAIT) {
/*
            printf("timeout %d\n", time2-time);
*/
            handle_timeout();
            time = time2;
        }
        /*
                printf("%d\n", timer.tv_usec);
         */
        if (FD_ISSET(fileno(stdin), &mask)) {
            handle_stdin();
        }
        check_ports(&mask);
    }
}

int main(int argc, char**argv) {
    init(argc, argv);
    loop();
    return EXIT_SUCCESS;
}
