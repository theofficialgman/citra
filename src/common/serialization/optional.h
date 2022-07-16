// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <boost/config.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/library_version_type.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>

namespace boost {
namespace serialization {

// Allow serialization of std::optional
template<class Archive, class T>
void save(Archive& ar, const std::optional<T>& t, const unsigned int /*version*/) {
    const bool tflag = t.has_value();
    ar << boost::serialization::make_nvp("initialized", tflag);
    if (tflag) {
        ar << boost::serialization::make_nvp("value", *t);
    }
}

template<class Archive, class T>
void load(Archive& ar, std::optional<T>& t, const unsigned int version) {
    bool tflag;
    ar >> boost::serialization::make_nvp("initialized", tflag);
    if (!tflag) {
        t.reset();
        return;
    }

    if (version == 0) {
        boost::serialization::item_version_type item_version(0);
        boost::serialization::library_version_type library_version(
            ar.get_library_version()
        );

        if (boost::serialization::library_version_type(3) < library_version) {
            ar >> BOOST_SERIALIZATION_NVP(item_version);
        }
    }

    if (!t.has_value()) {
        t = T();
    }

    ar >> boost::serialization::make_nvp("value", *t);
}

template<class Archive, class T>
void serialize(Archive& ar, std::optional<T>& t, const unsigned int version) {
    boost::serialization::split_free(ar, t, version);
}

template<class T>
struct version<std::optional<T>> {
    BOOST_STATIC_CONSTANT(int, value = 1);
};

} // serialization
} // boost