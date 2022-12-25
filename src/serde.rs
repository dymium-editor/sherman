//! `serde` support for [`RleTree`]s

use serde::de::{self, Deserialize, Deserializer, Visitor};
use serde::{Serialize, Serializer};

use std::fmt;
use std::marker::PhantomData;

use crate::param::{RleTreeConfig, SupportsInsert};
use crate::{Index, RleTree, Slice};

impl<I, S, P, const M: usize> Serialize for RleTree<I, S, P, M>
where
    I: Serialize + Index,
    S: Serialize + Slice<I>,
    P: RleTreeConfig<I, S, M>,
{
    fn serialize<Se: Serializer>(&self, serializer: Se) -> Result<Se::Ok, Se::Error> {
        serializer.collect_seq(self.iter(..).map(|entry| (entry.size(), entry.slice())))
    }
}

impl<'de, I, S, P, const M: usize> Deserialize<'de> for RleTree<I, S, P, M>
where
    I: Deserialize<'de> + Index,
    S: Deserialize<'de> + Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(RleTreeVisitor { marker: PhantomData })
    }
}

struct RleTreeVisitor<I, S, P: RleTreeConfig<I, S, M>, const M: usize> {
    marker: PhantomData<RleTree<I, S, P, M>>,
}

impl<'de, I, S, P, const M: usize> Visitor<'de> for RleTreeVisitor<I, S, P, M>
where
    I: Deserialize<'de> + Index,
    S: Deserialize<'de> + Slice<I>,
    P: RleTreeConfig<I, S, M> + SupportsInsert<I, S, M>,
{
    type Value = RleTree<I, S, P, M>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sequence of (size, value) pairs")
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut this = RleTree::new_empty();
        while let Some((size, slice)) = seq.next_element()? {
            let idx = this.size();
            this.insert(idx, slice, size);
        }

        Ok(this)
    }
}
