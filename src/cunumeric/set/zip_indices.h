namespace cunumeric {

using namespace legate;

template <typename VAL>
struct ZippedIndex {
  VAL value;
  int64_t index;
};

// Surprisingly it seems as though thrust can't figure out this comparison
template <typename VAL>
struct ZippedComparator {
  bool operator()(const ZippedIndex<VAL>& a, const ZippedIndex<VAL>& b)
  {
    return (a.value == b.value) ? a.index < b.index : a.value < b.value;
  }
};

inline int64_t rowwise_linearize(int32_t DIM, const DomainPoint& p, const DomainPoint& parent_point)
{
  int multiplier = 1;
  int64_t index  = 0;
  for (int i = DIM - 1; i >= 0; i--) {
    index += p[i] * multiplier;
    multiplier *= parent_point[i];
  }

  return index;
}

}  // namespace cunumeric
