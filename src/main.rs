use std::{iter, sync::Barrier, time};

use crossbeam_utils::{thread::scope, CachePadded};

fn main() {
    let mut args = std::env::args()
        .skip(1)
        .map(|it| it.parse::<u32>().unwrap());

    let options = Options {
        n_threads: args.next().unwrap(),
        n_locks: args.next().unwrap(),
        n_ops: args.next().unwrap(),
        n_rounds: args.next().unwrap(),
    };
    println!("{:#?}\n", options);

    bench::<mutexes::Std>(&options);
    bench::<mutexes::ParkingLot>(&options);
    bench::<mutexes::Spin>(&options);
    bench::<mutexes::AmdSpin>(&options);
    bench::<mutexes::TicketSpin>(&options);

    println!();
    bench::<mutexes::Std>(&options);
    bench::<mutexes::ParkingLot>(&options);
    bench::<mutexes::Spin>(&options);
    bench::<mutexes::AmdSpin>(&options);
    bench::<mutexes::TicketSpin>(&options);
}

fn bench<M: Mutex>(options: &Options) {
    let mut times = (0..options.n_rounds)
        .map(|_| run_bench::<M>(options))
        .collect::<Vec<_>>();
    times.sort();

    let avg = times.iter().sum::<time::Duration>() / options.n_rounds;
    let min = times[0];
    let max = *times.last().unwrap();

    let avg = format!("{:?}", avg);
    let min = format!("{:?}", min);
    let max = format!("{:?}", max);

    println!(
        "{:<20} avg {:<12} min {:<12} max {:<12}",
        M::LABEL,
        avg,
        min,
        max
    )
}

#[derive(Debug)]
struct Options {
    n_threads: u32,
    n_locks: u32,
    n_ops: u32,
    n_rounds: u32,
}

fn random_numbers(seed: u32) -> impl Iterator<Item = u32> {
    let mut random = seed;
    iter::repeat_with(move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    })
}

trait Mutex: Sync + Send + Default {
    const LABEL: &'static str;
    fn with_lock(&self, f: impl FnOnce(&mut u32));
}

fn run_bench<M: Mutex>(options: &Options) -> time::Duration {
    let locks = &(0..options.n_locks)
        .map(|_| CachePadded::new(M::default()))
        .collect::<Vec<_>>();

    let start_barrier = &Barrier::new(options.n_threads as usize + 1);
    let end_barrier = &Barrier::new(options.n_threads as usize + 1);

    let elapsed = scope(|scope| {
        let thread_seeds = random_numbers(0x6F4A955E).scan(0x9BA2BF27, |state, n| {
            *state ^= n;
            Some(*state)
        });
        for thread_seed in thread_seeds.take(options.n_threads as usize) {
            scope.spawn(move |_| {
                start_barrier.wait();
                let indexes = random_numbers(thread_seed)
                    .map(|it| it % options.n_locks)
                    .map(|it| it as usize)
                    .take(options.n_ops as usize);
                for idx in indexes {
                    locks[0].with_lock(|cnt| *cnt += 1);
                }
                end_barrier.wait();
            });
        }

        std::thread::sleep(time::Duration::from_millis(100));
        start_barrier.wait();
        let start = time::Instant::now();
        end_barrier.wait();
        let elapsed = start.elapsed();

        let mut total = 0;
        for lock in locks.iter() {
            lock.with_lock(|cnt| total += *cnt);
        }
        assert_eq!(total, options.n_threads * options.n_ops);

        elapsed
    })
    .unwrap();
    elapsed
}

mod mutexes {
    use super::Mutex;

    pub(crate) type Std = std::sync::Mutex<u32>;
    impl Mutex for Std {
        const LABEL: &'static str = "std::sync::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock().unwrap();
            f(&mut guard)
        }
    }

    pub(crate) type ParkingLot = parking_lot::Mutex<u32>;
    impl Mutex for ParkingLot {
        const LABEL: &'static str = "parking_lot::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }

    pub(crate) type Spin = spin::Mutex<u32>;
    impl Mutex for Spin {
        const LABEL: &'static str = "spin::Mutex";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }

    pub(crate) type TicketSpin = crate::ticket_spinlock::TicketSpinLock<u32>;
    impl Mutex for TicketSpin {
        const LABEL: &'static str = "TicketSpin";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }

    pub(crate) type AmdSpin = crate::amd_spinlock::AmdSpinlock<u32>;
    impl Mutex for AmdSpin {
        const LABEL: &'static str = "AmdSpinlock";
        fn with_lock(&self, f: impl FnOnce(&mut u32)) {
            let mut guard = self.lock();
            f(&mut guard)
        }
    }
}

mod amd_spinlock {
    use std::{
        cell::UnsafeCell,
        ops,
        sync::atomic::{spin_loop_hint, AtomicBool, Ordering},
    };

    #[derive(Default)]
    pub(crate) struct AmdSpinlock<T> {
        locked: AtomicBool,
        data: UnsafeCell<T>,
    }
    unsafe impl<T: Send> Send for AmdSpinlock<T> {}
    unsafe impl<T: Send> Sync for AmdSpinlock<T> {}

    pub(crate) struct AmdSpinlockGuard<'a, T> {
        lock: &'a AmdSpinlock<T>,
    }

    impl<T> AmdSpinlock<T> {
        pub(crate) fn lock(&self) -> AmdSpinlockGuard<T> {
            loop {
                let was_locked = self.locked.load(Ordering::Relaxed);
                if !was_locked
                    && self
                        .locked
                        .compare_exchange_weak(
                            was_locked,
                            true,
                            Ordering::Acquire,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    break;
                }
                spin_loop_hint()
            }
            AmdSpinlockGuard { lock: self }
        }
    }

    impl<'a, T> ops::Deref for AmdSpinlockGuard<'a, T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &*ptr }
        }
    }

    impl<'a, T> ops::DerefMut for AmdSpinlockGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &mut *ptr }
        }
    }

    impl<'a, T> Drop for AmdSpinlockGuard<'a, T> {
        fn drop(&mut self) {
            self.lock.locked.store(false, Ordering::Release)
        }
    }
}

mod ticket_spinlock {
    use std::{
        cell::UnsafeCell,
        ops,
        sync::atomic::{spin_loop_hint, AtomicUsize, Ordering},
    };

    //#[repr(align(64))]
    #[derive(Default)]
    struct CacheAligned(AtomicUsize);

    #[derive(Default)]
    pub(crate) struct TicketSpinLock<T> {
        acq: CacheAligned,
        rel: CacheAligned,
        data: UnsafeCell<T>,
    }
    unsafe impl<T: Send> Send for TicketSpinLock<T> {}
    unsafe impl<T: Send> Sync for TicketSpinLock<T> {}

    pub(crate) struct AmdSpinlockGuard<'a, T> {
        lock: &'a TicketSpinLock<T>,
    }

    impl<T> TicketSpinLock<T> {
        pub(crate) fn lock(&self) -> AmdSpinlockGuard<T> {
            let my_turn = self.acq.0.fetch_add(1, Ordering::AcqRel);
            while my_turn != self.rel.0.load(Ordering::Acquire) {
                spin_loop_hint()
            }
            AmdSpinlockGuard { lock: self }
        }
    }

    impl<'a, T> ops::Deref for AmdSpinlockGuard<'a, T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &*ptr }
        }
    }

    impl<'a, T> ops::DerefMut for AmdSpinlockGuard<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let ptr = self.lock.data.get();
            unsafe { &mut *ptr }
        }
    }

    impl<'a, T> Drop for AmdSpinlockGuard<'a, T> {
        fn drop(&mut self) {
            self.lock.rel.0.fetch_add(1, Ordering::AcqRel);
        }
    }
}
