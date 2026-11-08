/* See n64tts.h. Integer only: a test in the host tool forbids floating-point type names here. */
#include "n64tts.h"

#define ORDER N64TTS_ORDER
#define HALF (ORDER / 2)
#define SUB 80 /* samples between filter updates: 5 ms */
#define FRAME_BYTES N64TTS_FRAME_BYTES
#define ENTRY_BYTES 10

#define SEAM 960 /* samples over which neighbouring joins are blended: 60 ms */
#define SEAM_MIN_GAIN 70 /* below this a frame is near silence, with no shape worth matching */

typedef long long i64;

/* 2^(k/16) in Q15 */
static const i64 pow2_table[16] = {32768, 34219, 35734, 37316, 38968, 40693, 42495, 44376,
                                   46341, 48393, 50535, 52773, 55109, 57549, 60097, 62757};

static int read_s16(const unsigned char *p)
{
    int v = p[0] | (p[1] << 8);
    return v >= 32768 ? v - 65536 : v;
}

static unsigned int read_u16(const unsigned char *p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8);
}

static unsigned int read_u32(const unsigned char *p)
{
    return read_u16(p) | (read_u16(p + 2) << 16);
}

int n64tts_voice_init(n64tts_voice *v, const unsigned char *data, unsigned int length)
{
    unsigned int at = 5, i, phones;
    if (length < 5 || data[0] != 'C' || data[1] != 'V' || data[2] != 'O' || data[3] != 'X')
        return -1;
    phones = data[4];
    for (i = 0; i < phones; i++) {
        if (at >= length)
            return -1;
        at += 2u + data[at]; /* length, name, duration scale */
    }
    if (at + 257u * 2u + 128u * 2u + 2u > length)
        return -1;
    v->cos_table = data + at;
    at += 257u * 2u;
    v->pulse = data + at;
    at += 128u * 2u;
    v->joins = read_u16(data + at);
    at += 2;
    v->index = data + at;
    if (at + v->joins * ENTRY_BYTES > length)
        return -1;
    at += v->joins * ENTRY_BYTES;
    v->frames = data + at;
    v->frame_count = (length - at) / FRAME_BYTES;
    return 0;
}

typedef struct {
    unsigned int offset;
    unsigned int frames;
} half_t;

/* The first entry at or after (a, b). */
static unsigned int lower_bound(const n64tts_voice *v, unsigned int key)
{
    unsigned int lo = 0, hi = v->joins;
    while (lo < hi) {
        unsigned int mid = lo + (hi - lo) / 2;
        const unsigned char *e = v->index + mid * ENTRY_BYTES;
        if ((((unsigned int)e[0] << 8) | e[1]) < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

/* Find the frames joining a to b. A join never recorded borrows its halves from other joins. */
static void find_halves(const n64tts_voice *v, unsigned int a, unsigned int b, half_t *left,
                        half_t *right, n64tts_stats *stats)
{
    unsigned int i = lower_bound(v, (a << 8) | b);
    const unsigned char *e;
    left->offset = left->frames = right->offset = right->frames = 0;
    stats->joins++;
    if (i < v->joins) {
        e = v->index + i * ENTRY_BYTES;
        if (e[0] == a && e[1] == b) {
            unsigned int join = read_u16(e + 8);
            left->offset = read_u32(e + 2);
            left->frames = join;
            right->offset = left->offset + join;
            right->frames = read_u16(e + 6) - join;
            return;
        }
    }
    stats->borrowed_joins++;
    i = lower_bound(v, a << 8);
    if (i < v->joins && v->index[i * ENTRY_BYTES] == a) {
        e = v->index + i * ENTRY_BYTES;
        left->offset = read_u32(e + 2);
        left->frames = read_u16(e + 8);
    }
    for (i = 0; i < v->joins; i++) {
        e = v->index + i * ENTRY_BYTES;
        if (e[1] == b) {
            unsigned int join = read_u16(e + 8);
            right->offset = read_u32(e + 2) + join;
            right->frames = read_u16(e + 6) - join;
            return;
        }
    }
}

unsigned int n64tts_output_len(const n64tts_event *events, unsigned int count)
{
    unsigned int total = 0, i, last;
    if (count < 2)
        return 0;
    for (i = 0; i < count; i++)
        total += events[i].samples;
    last = events[count - 1].samples;
    return total - events[0].samples / 2 - (last - last / 2);
}

static i64 cos_q15(const n64tts_voice *v, unsigned int w)
{
    unsigned int i = w >> 8;
    i64 frac = (i64)(w & 255u);
    i64 a = read_s16(v->cos_table + i * 2);
    i64 b = read_s16(v->cos_table + (i + 1) * 2);
    return a + (((b - a) * frac) >> 8);
}

/* One half of the filter polynomial, from every other cosine: Q24. */
static void build_poly(const i64 *q, i64 *f)
{
    int i, j;
    f[0] = 16777216;
    f[1] = -(q[0] * 1024);
    for (i = 2; i <= HALF; i++) {
        i64 b = q[2 * (i - 1)];
        f[i] = 2 * f[i - 2] - ((b * f[i - 1]) >> 14);
        for (j = i - 1; j >= 2; j--)
            f[j] += f[j - 2] - ((b * f[j - 1]) >> 14);
        f[1] -= b * 1024;
    }
}

/* Line spectral frequencies (0..65535 is 0..pi) to direct-form coefficients in Q24. */
static void lsf_to_filter(const n64tts_voice *v, const unsigned int *lsf, i64 *a)
{
    i64 q[ORDER], f1[HALF + 1], f2[HALF + 1];
    int k;
    for (k = 0; k < ORDER; k++)
        q[k] = cos_q15(v, lsf[k]);
    build_poly(q, f1);
    build_poly(q + 1, f2);
    for (k = 1; k <= HALF; k++) {
        i64 p = f1[k] + f1[k - 1];
        i64 m = f2[k] - f2[k - 1];
        a[k - 1] = (p + m) >> 1;
        a[ORDER - k] = (p - m) >> 1;
    }
}

static i64 gain_from(unsigned int q8)
{
    unsigned int index = q8 >> 8;
    return (pow2_table[index % 16] << (index / 16)) >> 15;
}

static i64 next_noise(unsigned int *rng)
{
    /* xorshift32: the same every run */
    unsigned int x = *rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *rng = x;
    return (i64)(x >> 16) - 32768;
}

static i64 clamp64(i64 v, i64 lo, i64 hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static unsigned int min_u(unsigned int a, unsigned int b)
{
    return a < b ? a : b;
}

/* Move a frame toward its neighbour across a seam: half the difference at the seam itself, so
 * both sides arrive at the same value, fading to nothing `width` samples away. */
static void blend_seam(const unsigned char *mine, const unsigned char *other, unsigned int distance,
                       unsigned int width, unsigned int *lsf, unsigned int *g)
{
    int k, weight, value;
    if (!mine || !other || distance >= width)
        return;
    if (mine[ORDER] < SEAM_MIN_GAIN || other[ORDER] < SEAM_MIN_GAIN)
        return;
    weight = (int)(((width - distance) * 128u) / width);
    for (k = 0; k < ORDER; k++) {
        value = (int)lsf[k] + ((int)other[k] - (int)mine[k]) * weight;
        lsf[k] = value < 0 ? 0u : (unsigned int)value;
    }
    value = (int)*g + ((int)other[ORDER] - (int)mine[ORDER]) * weight;
    *g = value < 0 ? 0u : (unsigned int)value;
}

static const unsigned char *frame_at(const n64tts_voice *v, const half_t *h, unsigned int i)
{
    return h->frames ? v->frames + (h->offset + i) * FRAME_BYTES : 0;
}

void n64tts_render(const n64tts_voice *v, const n64tts_event *events, unsigned int count,
                   const n64tts_params *params, short *out, n64tts_stats *stats)
{
    i64 a[ORDER], history[ORDER];
    unsigned int lsf[ORDER], interp[ORDER];
    half_t left, right;
    const unsigned char *prev_end = 0, *this_start = 0, *this_end = 0, *next_start = 0;
    unsigned int n = n64tts_output_len(events, count);
    unsigned int pair = 0, pair_start = 0, half_a = 1, half_b = 1, pos = 0;
    unsigned int head = 0, phase = 0, rng = 0x2545F491u;
    i64 period_scale = 4096, prev_source = 0, emphasis = 0;
    int have = 0, k;

    stats->samples = n;
    stats->filter_updates = 0;
    stats->joins = 0;
    stats->borrowed_joins = 0;
    if (n == 0)
        return;
    left.offset = left.frames = right.offset = right.frames = 0;
    for (k = 0; k < ORDER; k++) {
        a[k] = 0;
        history[k] = 0;
    }

    while (pos < n) {
        const n64tts_event *ea, *eb;
        unsigned int at, end, g = 0, voicing = 0, f0, span;
        half_t side;
        unsigned int place;
        i64 inc, amp, voiced, unvoiced;

        /* which join are we in? */
        for (;;) {
            unsigned int ha, hb;
            ea = &events[pair];
            eb = &events[pair + 1];
            ha = ea->samples - ea->samples / 2;
            hb = eb->samples / 2;
            if (!have) {
                find_halves(v, ea->phone, eb->phone, &left, &right, stats);
                prev_end = this_end;
                this_start = frame_at(v, &left, 0);
                this_end = right.frames ? frame_at(v, &right, right.frames - 1) : 0;
                next_start = 0;
                if (pair + 2 < count) {
                    half_t next_left, next_right;
                    n64tts_stats ignored;
                    find_halves(v, eb->phone, events[pair + 2].phone, &next_left, &next_right, &ignored);
                    next_start = frame_at(v, &next_left, 0);
                }
                half_a = ha ? ha : 1;
                half_b = hb ? hb : 1;
                have = 1;
            }
            if (pos < pair_start + ha + hb || pair + 2 >= count)
                break;
            pair_start += ha + hb;
            pair++;
            have = 0;
        }
        at = pos - pair_start;

        /* where in the stored frames, in Q8 */
        if (at < half_a) {
            side = left;
            place = (unsigned int)((((i64)left.frames << 8) * at) / half_a);
        } else {
            side = right;
            place = (unsigned int)((((i64)right.frames << 8) * min_u(at - half_a, half_b - 1)) / half_b);
        }
        if (side.frames > 0) {
            unsigned int i = min_u(place >> 8, side.frames - 1);
            unsigned int j = min_u(i + 1, side.frames - 1);
            unsigned int frac = place & 255u, floor_w = 256;
            const unsigned char *f0p = v->frames + (side.offset + i) * FRAME_BYTES;
            const unsigned char *f1p = v->frames + (side.offset + j) * FRAME_BYTES;
            for (k = 0; k < ORDER; k++)
                interp[k] = f0p[k] * (256u - frac) + f1p[k] * frac + 128u; /* a byte is (b + 0.5) steps */
            g = f0p[ORDER] * (256u - frac) + f1p[ORDER] * frac;
            voicing = (f0p[ORDER + 1] * (256u - frac) + f1p[ORDER + 1] * frac) >> 8;
            if (at < half_a)
                blend_seam(this_start, prev_end, at, min_u(SEAM, half_a), interp, &g);
            else
                blend_seam(this_end, next_start, half_a + half_b - 1 - at, min_u(SEAM, half_b), interp, &g);
            /* a different vocal tract length moves every resonance by the same ratio */
            for (k = 0; k < ORDER; k++) {
                unsigned int w = (unsigned int)clamp64(((i64)interp[k] * params->size) >> 12, 256, 65000);
                lsf[k] = w > floor_w ? w : floor_w;
                floor_w = lsf[k] + 256u;
            }
            for (k = ORDER - 1; k >= 0; k--)
                lsf[k] = min_u(lsf[k], 65200u - (unsigned int)(ORDER - 1 - k) * 256u);
            lsf_to_filter(v, lsf, a);
            stats->filter_updates++;
        }

        /* pitch: straight lines between the middles of the sounds */
        span = half_a + half_b;
        f0 = (unsigned int)(((i64)ea->f0 * (span - at) + (i64)eb->f0 * at) / span);
        inc = (((i64)f0) * 4096) / N64TTS_OUT_RATE; /* phase in Q16 per sample */
        amp = gain_from(g);
        voiced = (((i64)voicing * 273) * (4096 - params->breath)) >> 12;
        if (voiced > 4096)
            voiced = 4096;
        unvoiced = 4096 - voiced;

        end = min_u(pos + SUB, n);
        while (pos < end) {
            i64 p, noise, mixed, excitation, acc, y;
            phase += (unsigned int)((inc * period_scale) >> 12);
            if (phase >= 65536u) {
                phase -= 65536u;
                period_scale = clamp64(4096 + ((next_noise(&rng) * params->rough) >> 18), 2048, 8192);
            }
            p = read_s16(v->pulse + (phase >> 9) * 2);
            noise = (next_noise(&rng) * amp * 7094) >> 27;
            mixed = (((p * amp) >> 12) * voiced + noise * unvoiced) >> 12;
            excitation = mixed - ((params->bright * prev_source) >> 15);
            prev_source = mixed;
            acc = excitation * 16777216;
            for (k = 0; k < ORDER; k++)
                acc -= a[k] * history[(head + (unsigned int)k) % ORDER];
            y = clamp64(acc >> 24, -1048576, 1048576);
            head = (head + ORDER - 1) % ORDER;
            history[head] = y;
            emphasis = clamp64(y + ((31130 * emphasis) >> 15), -4194304, 4194304);
            out[pos] = (short)clamp64(emphasis, -32768, 32767);
            pos++;
        }
    }
}
