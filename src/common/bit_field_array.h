#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include "common/swap.h"

// Language limitations require the following to make these formattable
// (formatter<BitFieldArray<position, bits, size, T>::Ref> is not legal)
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayConstRef;
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayRef;
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayConstIterator;
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayIterator;

#pragma pack(1)
template <std::size_t position, std::size_t bits, std::size_t size, typename T,
          // StorageType is T for non-enum types and the underlying type of T if
          // T is an enumeration. Note that T is wrapped within an enable_if in the
          // former case to workaround compile errors which arise when using
          // std::underlying_type<T>::type directly.
          typename StorageType = typename std::conditional_t<
              std::is_enum<T>::value, std::underlying_type<T>, std::enable_if<true, T>>::type>
struct BitFieldArray
{
  using Ref = BitFieldArrayRef<position, bits, size, T, StorageType>;
  using ConstRef = BitFieldArrayConstRef<position, bits, size, T, StorageType>;
  using Iterator = BitFieldArrayIterator<position, bits, size, T, StorageType>;
  using ConstIterator = BitFieldArrayConstIterator<position, bits, size, T, StorageType>;

private:
  // This constructor might be considered ambiguous:
  // Would it initialize the storage or just the bitfield?
  // Hence, delete it. Use the assignment operator to set bitfield values!
  BitFieldArray(T val) = delete;

public:
  // Force default constructor to be created
  // so that we can use this within unions
  constexpr BitFieldArray() = default;

  // Initializer list constructor
  constexpr BitFieldArray(std::initializer_list<T> items) : storage(StorageType{}) {
      u32 index = 0;
      for (auto& item : items) {
        SetValue(index++, item);
    }
  }

  // We explicitly delete the copy assignment operator here, because the
  // default copy assignment would copy the full storage value, rather than
  // just the bits relevant to this particular bit field.
  // Ideally, we would just implement the copy assignment to copy only the
  // relevant bits, but we're prevented from doing that because the savestate
  // code expects that this class is trivially copyable.
  BitFieldArray& operator=(const BitFieldArray&) = delete;

public:
  constexpr bool IsSigned() const { return std::is_signed<T>(); }
  constexpr std::size_t StartBit() const { return position; }
  constexpr std::size_t NumBits() const { return bits; }
  constexpr std::size_t Size() const { return size; }
  constexpr std::size_t TotalNumBits() const { return bits * size; }

  constexpr T Value(size_t index) const { return Value(std::is_signed<T>(), index); }
  constexpr void SetValue(size_t index, T value) {
    const size_t pos = position + bits * index;
    storage = (storage & ~GetElementMask(index)) |
              ((static_cast<StorageType>(value) << pos) & GetElementMask(index));
  }
  Ref operator[](size_t index) { return Ref(this, index); }
  constexpr const ConstRef operator[](size_t index) const { return ConstRef(this, index); }

  constexpr Iterator begin() { return Iterator(this, 0); }
  constexpr Iterator end() { return Iterator(this, size); }
  constexpr ConstIterator begin() const { return ConstIterator(this, 0); }
  constexpr ConstIterator end() const { return ConstIterator(this, size); }
  constexpr ConstIterator cbegin() const { return begin(); }
  constexpr ConstIterator cend() const { return end(); }

private:
  // Unsigned version of StorageType
  using StorageTypeU = std::make_unsigned_t<StorageType>;

  constexpr T Value(std::true_type, size_t index) const
  {
    const size_t pos = position + bits * index;
    const size_t shift_amount = 8 * sizeof(StorageType) - bits;
    return static_cast<T>((storage << (shift_amount - pos)) >> shift_amount);
  }

  constexpr T Value(std::false_type, size_t index) const
  {
    const size_t pos = position + bits * index;
    return static_cast<T>((storage & GetElementMask(index)) >> pos);
  }

  static constexpr StorageType GetElementMask(size_t index)
  {
    const size_t pos = position + bits * index;
    return (std::numeric_limits<StorageTypeU>::max() >> (8 * sizeof(StorageType) - bits)) << pos;
  }

  StorageType storage;

  static_assert(bits * size + position <= 8 * sizeof(StorageType), "Bitfield array out of range");
  static_assert(sizeof(T) <= sizeof(StorageType), "T must fit in StorageType");

  // And, you know, just in case people specify something stupid like bits=position=0x80000000
  static_assert(position < 8 * sizeof(StorageType), "Invalid position");
  static_assert(bits <= 8 * sizeof(T), "Invalid number of bits");
  static_assert(bits > 0, "Invalid number of bits");
  static_assert(size <= 8 * sizeof(StorageType), "Invalid size");
  static_assert(size > 0, "Invalid size");
};
#pragma pack()

template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayConstRef
{
  friend struct BitFieldArray<position, bits, size, T, S>;
  friend class BitFieldArrayConstIterator<position, bits, size, T, S>;

public:
  constexpr T Value() const { return m_array->Value(m_index); };
  constexpr operator T() const { return Value(); }

private:
  constexpr BitFieldArrayConstRef(const BitFieldArray<position, bits, size, T, S>* array,
                                  size_t index)
      : m_array(array), m_index(index)
  {
  }

  const BitFieldArray<position, bits, size, T, S>* const m_array;
  const size_t m_index;
};

template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayRef
{
  friend struct BitFieldArray<position, bits, size, T, S>;
  friend class BitFieldArrayIterator<position, bits, size, T, S>;

public:
  constexpr T Value() const { return m_array->Value(m_index); };
  constexpr operator T() const { return Value(); }
  T operator=(const BitFieldArrayRef<position, bits, size, T, S>& value) const
  {
    m_array->SetValue(m_index, value);
    return value;
  }
  T operator=(T value) const
  {
    m_array->SetValue(m_index, value);
    return value;
  }

private:
  constexpr BitFieldArrayRef(BitFieldArray<position, bits, size, T, S>* array, size_t index)
      : m_array(array), m_index(index)
  {
  }

  BitFieldArray<position, bits, size, T, S>* const m_array;
  const size_t m_index;
};

// Satisfies LegacyOutputIterator / std::output_iterator.
// Does not satisfy LegacyInputIterator / std::input_iterator as std::output_iterator_tag does not
// extend std::input_iterator_tag.
// Does not satisfy LegacyForwardIterator / std::forward_iterator, as that requires use of real
// references instead of proxy objects.
// This iterator allows use of BitFieldArray in range-based for loops, and with fmt::join.
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayIterator
{
  friend struct BitFieldArray<position, bits, size, T, S>;

public:
  using iterator_category = std::output_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using pointer = void;
  using reference = BitFieldArrayRef<position, bits, size, T, S>;

private:
  constexpr BitFieldArrayIterator(BitFieldArray<position, bits, size, T, S>* array, size_t index)
      : m_array(array), m_index(index)
  {
  }

public:
  // Required by std::input_or_output_iterator
  constexpr BitFieldArrayIterator() = default;
  // Required by LegacyIterator
  constexpr BitFieldArrayIterator(const BitFieldArrayIterator& other) = default;
  // Required by LegacyIterator
  BitFieldArrayIterator& operator=(const BitFieldArrayIterator& other) = default;
  // Move constructor and assignment operators, explicitly defined for completeness
  constexpr BitFieldArrayIterator(BitFieldArrayIterator&& other) = default;
  BitFieldArrayIterator& operator=(BitFieldArrayIterator&& other) = default;

public:
  BitFieldArrayIterator& operator++()
  {
    m_index++;
    return *this;
  }
  BitFieldArrayIterator operator++(int)
  {
    BitFieldArrayIterator other(*this);
    ++*this;
    return other;
  }
  constexpr reference operator*() const { return reference(m_array, m_index); }
  constexpr bool operator==(BitFieldArrayIterator other) const { return m_index == other.m_index; }
  constexpr bool operator!=(BitFieldArrayIterator other) const { return m_index != other.m_index; }

private:
  BitFieldArray<position, bits, size, T, S>* m_array;
  size_t m_index;
};

// Satisfies LegacyInputIterator / std::input_iterator.
// Does not satisfy LegacyForwardIterator / std::forward_iterator, as that requires use of real
// references instead of proxy objects.
// This iterator allows use of BitFieldArray in range-based for loops, and with fmt::join.
template <std::size_t position, std::size_t bits, std::size_t size, typename T, typename S>
class BitFieldArrayConstIterator
{
  friend struct BitFieldArray<position, bits, size, T, S>;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type = T;
  using difference_type = ptrdiff_t;
  using pointer = void;
  using reference = BitFieldArrayConstRef<position, bits, size, T, S>;

private:
  constexpr BitFieldArrayConstIterator(const BitFieldArray<position, bits, size, T, S>* array,
                                       size_t index)
      : m_array(array), m_index(index)
  {
  }

public:
  // Required by std::input_or_output_iterator
  constexpr BitFieldArrayConstIterator() = default;
  // Required by LegacyIterator
  constexpr BitFieldArrayConstIterator(const BitFieldArrayConstIterator& other) = default;
  // Required by LegacyIterator
  BitFieldArrayConstIterator& operator=(const BitFieldArrayConstIterator& other) = default;
  // Move constructor and assignment operators, explicitly defined for completeness
  constexpr BitFieldArrayConstIterator(BitFieldArrayConstIterator&& other) = default;
  BitFieldArrayConstIterator& operator=(BitFieldArrayConstIterator&& other) = default;

public:
  BitFieldArrayConstIterator& operator++()
  {
    m_index++;
    return *this;
  }
  BitFieldArrayConstIterator operator++(int)
  {
    BitFieldArrayConstIterator other(*this);
    ++*this;
    return other;
  }
  constexpr reference operator*() const { return reference(m_array, m_index); }
  constexpr bool operator==(BitFieldArrayConstIterator other) const
  {
    return m_index == other.m_index;
  }
  constexpr bool operator!=(BitFieldArrayConstIterator other) const
  {
    return m_index != other.m_index;
  }

private:
  const BitFieldArray<position, bits, size, T, S>* m_array;
  size_t m_index;
};
