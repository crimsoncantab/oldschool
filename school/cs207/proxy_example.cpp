/*
  This file demonstrates the Proxy design pattern.
  It contains two classes: a "SimpleSet" and a "SimpleElement".

  When the SimpleSet changes size, all of its elements are copied to new
  locations in memory and renumbered. But that's OK: since the SimpleElements
  are proxies, they always access the SimpleSet using the most up-to-date
  memory address and index.

  We use dynamic memory allocation here to make the issues obvious; you would
  almost certainly use an STL container. Our proxy is also slow: it takes O(N)
  time to access an element, where N is the number of elements. You will fix
  this in your Graph.
*/

#include <iostream>
#include <assert.h>

class SimpleSet {
  // Internal type for set elements
  struct internal_element {
    const char *text;
    int uid;
  };

 public:
  SimpleSet()
    : elements_(), size_(0), next_uid_(0) {
  }
  ~SimpleSet() {
    delete[] elements_;
  }

  /** The proxy class */
  class SimpleElement {
   public:
    SimpleElement() {
    }
    const char *text() const {
      return fetch().text;
    }
    void set_text(const char *text) {
      fetch().text = text;
    }
   private:
    SimpleSet *set_;
    size_t uid_;
    SimpleElement(SimpleSet *set, size_t uid)
      : set_(set), uid_(uid) {
    }
    internal_element& fetch() const {
      for (size_t i = 0; i < set_->size(); ++i)
	if (set_->elements_[i].uid == uid_)
	  return set_->elements_[i];
      assert(0);
    }
    friend class SimpleSet;	// allow SimpleSet to access private parts
  };

  /** Return SimpleSet's size. */
  size_t size() const {
    return size_;
  }
  /** Return a proxy for element @a i. */
  SimpleElement operator[](size_t i) {
    assert(i < size());
    return SimpleElement(this, i);
  }
  /** Add a new element at the end. */
  SimpleElement push_back(const char* text) {
    internal_element* new_elements = new internal_element[size_ + 1];
    for (size_t i = 0; i < size_; ++i)
      new_elements[i] = elements_[i];
    new_elements[size_].text = text;
    new_elements[size_].uid = next_uid_;
    delete[] elements_;
    elements_ = new_elements;
    ++size_;
    ++next_uid_;
    return SimpleElement(this, next_uid_ - 1);
  }
  /** Remove the element at position @a i, moving later elements down. */
  void remove(size_t i) {
    assert(i < size());
    for (++i; i < size(); ++i)
      elements_[i - 1] = elements_[i];
    --size_;
  }

 private:
  internal_element* elements_;
  size_t size_;
  size_t next_uid_;
  friend class SimpleElement;	// proxy needs permission to access
  				// private state
  SimpleSet(const SimpleSet&) = delete;
  SimpleSet& operator=(const SimpleSet&) = delete;
};

int main(int argc, char **argv) {
  SimpleSet v;
  SimpleSet::SimpleElement e0 = v.push_back("Hello");
  SimpleSet::SimpleElement e1 = v.push_back("World");
  std::cerr << e0.text() << " " << e1.text() << std::endl;
  // prints "Hello World"

  SimpleSet::SimpleElement e0_copy = v[0];
  e0.set_text("Goodbye");
  std::cerr << e0.text() << " " << e0_copy.text() << std::endl;
  // prints "Goodbye Goodbye": since SimpleElement is a proxy, e0 and e0_copy
  // both return the most up-to-date information

  SimpleSet::SimpleElement e2 = v.push_back("Friends");
  v.remove(1);
  std::cerr << e0.text() << " " << e2.text() << std::endl;
  // prints "Goodbye Friends": SimpleElement locates its element using a
  // unique number that stays stable even after SimpleSet's internal array
  // is rearranged
}
