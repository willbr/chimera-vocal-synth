/* n64tts: speech from filter tables, in integer maths, for small machines.
 *
 * C89 plus `long long`, which every N64 compiler provides. Nothing is allocated, the same input
 * gives the same samples on any machine, and voice data is read a byte at a time, so a voice built
 * on a little-endian PC works on the big-endian N64.
 *
 * Assumes `int` is 32 bits and that >> on a negative number is arithmetic. Both hold for the
 * compilers this targets. */
#ifndef N64TTS_H
#define N64TTS_H

#define N64TTS_ORDER 16
#define N64TTS_OUT_RATE 16000
#define N64TTS_FRAME_BYTES 18

typedef struct {
    const unsigned char *cos_table; /* 257 little-endian 16-bit values */
    const unsigned char *pulse;     /* 128 little-endian 16-bit values */
    const unsigned char *index;     /* 10-byte records sorted by (a, b) */
    const unsigned char *frames;
    unsigned int joins;
    unsigned int frame_count;
} n64tts_voice;

/* One sound to say. f0 is Hz * 16 at the middle of the sound. */
typedef struct {
    unsigned int samples;
    unsigned int f0;
    unsigned char phone;
} n64tts_event;

typedef struct {
    int size;   /* 4096 is unchanged; above is a smaller vocal tract, below a larger one */
    int breath; /* 0..4096: how much of the voiced source becomes noise */
    int bright; /* -32767..32767: tilt of the source, positive is brighter */
    int rough;  /* 0..4096: random variation of each pitch period */
} n64tts_params;

typedef struct {
    unsigned int samples;
    unsigned int filter_updates;
    unsigned int joins;
    unsigned int borrowed_joins;
} n64tts_stats;

/* Point `v` at a voice held in memory. Returns 0, or -1 if the data isn't a voice. The data is
 * only read, and must outlive `v`. */
int n64tts_voice_init(n64tts_voice *v, const unsigned char *data, unsigned int length);

/* How many samples n64tts_render will write for these events. */
unsigned int n64tts_output_len(const n64tts_event *events, unsigned int count);

/* Write 16 kHz mono samples into `out`, which must hold n64tts_output_len samples. */
void n64tts_render(const n64tts_voice *v, const n64tts_event *events, unsigned int count,
                   const n64tts_params *params, short *out, n64tts_stats *stats);

#endif
