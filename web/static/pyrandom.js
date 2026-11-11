// Bit-exact ports of the two random streams the composer uses, so a seed gives the same notes as the Python:
//   PyRandom    = CPython's `random` module (3.12): seed(int), random, getrandbits, randint, choice, uniform, sample.
//   NumpyRandom = NumPy's legacy global RandomState: seed(int) and randint(low, high) (masked rejection sampling).
// Both sit on MT19937. Checked against CPython and NumPy by pyrandom.test.mjs.

const N = 624, M = 397;

class MT19937 {
  constructor() { this.mt = new Uint32Array(N); this.mti = N + 1; }

  initGenrand(s) {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      // 1812433253 * p + i, modulo 2^32 (split to stay exact in doubles)
      mt[i] = (((((p & 0xffff0000) >>> 16) * 1812433253) << 16) + (p & 0x0000ffff) * 1812433253 + i) >>> 0;
    }
    this.mti = N;
  }

  initByArray(key) {
    const mt = this.mt;
    this.initGenrand(19650218);
    let i = 1, j = 0;
    for (let k = Math.max(N, key.length); k; k--) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ (((((p & 0xffff0000) >>> 16) * 1664525) << 16) + ((p & 0x0000ffff) * 1664525))) + key[j] + j) >>> 0;
      i++; j++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k; k--) {
      const p = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = ((mt[i] ^ (((((p & 0xffff0000) >>> 16) * 1566083941) << 16) + (p & 0x0000ffff) * 1566083941)) - i) >>> 0;
      i++;
      if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
    }
    mt[0] = 0x80000000;
    this.mti = N;
  }

  uint32() {
    const mt = this.mt;
    if (this.mti >= N) {
      let kk = 0, y;
      for (; kk < N - M; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      for (; kk < N - 1; kk++) {
        y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      y = (mt[N - 1] & 0x80000000) | (mt[0] & 0x7fffffff);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  // 53-bit float in [0, 1): identical formula in CPython (random_random) and NumPy (legacy_double).
  double() {
    const a = this.uint32() >>> 5, b = this.uint32() >>> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  }
}

const bitLength = n => (n === 0 ? 0 : 32 - Math.clz32(n));

export class PyRandom {
  constructor(seed = 0) { this.mt = new MT19937(); this.seed(seed); }

  // random.seed(n) for a non-negative int below 2^53: the key is |n| as little-endian 32-bit words.
  seed(n) {
    n = Math.abs(Math.trunc(n));
    const key = [];
    do { key.push(n % 4294967296); n = Math.floor(n / 4294967296); } while (n > 0);
    this.mt.initByArray(key);
  }

  random() { return this.mt.double(); }
  getrandbits(k) { return k === 0 ? 0 : this.mt.uint32() >>> (32 - k); }

  // _randbelow_with_getrandbits (only n < 2^32 is needed here)
  randbelow(n) {
    const k = bitLength(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  randint(a, b) {
    if (b + 1 - a <= 0) throw new Error(`empty range in randint(${a}, ${b})`);
    return a + this.randbelow(b + 1 - a);
  }

  choice(seq) {
    if (!seq.length) throw new Error('Cannot choose from an empty sequence');
    return seq[this.randbelow(seq.length)];
  }

  uniform(a, b) { return a + (b - a) * this.random(); }

  // random.sample(population, k), both the pool and the set branch of CPython's algorithm.
  sample(population, k) {
    const n = population.length;
    if (k < 0 || k > n) throw new Error('Sample larger than population or is negative');
    const result = new Array(k);
    let setsize = 21;
    if (k > 5) setsize += 4 ** Math.ceil(Math.log(k * 3) / Math.log(4));
    if (n <= setsize) {
      const pool = population.slice();
      for (let i = 0; i < k; i++) {
        const j = this.randbelow(n - i);
        result[i] = pool[j];
        pool[j] = pool[n - i - 1];
      }
    } else {
      const selected = new Set();
      for (let i = 0; i < k; i++) {
        let j = this.randbelow(n);
        while (selected.has(j)) j = this.randbelow(n);
        selected.add(j);
        result[i] = population[j];
      }
    }
    return result;
  }
}

export class NumpyRandom {
  constructor(seed = 0) { this.mt = new MT19937(); this.seed(seed); }

  // np.random.seed(n) for 0 <= n < 2^32 (legacy seeding: init_genrand).
  seed(n) { this.mt.initGenrand(n >>> 0); }

  // np.random.randint(low, high): int64 path, rng = high - low - 1 < 2^32, masked rejection on 32-bit draws.
  randint(low, high) {
    const rng = high - 1 - low;
    if (rng < 0) throw new Error('low >= high');
    if (rng === 0) return low;
    let mask = rng;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4; mask |= mask >>> 8; mask |= mask >>> 16;
    mask >>>= 0;
    let v;
    while ((v = (this.mt.uint32() & mask) >>> 0) > rng);
    return low + v;
  }

  random() { return this.mt.double(); }
}
