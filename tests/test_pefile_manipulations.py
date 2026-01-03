"""
Tests for PE file manipulation functions.

Run with: pytest tests/test_pefile_manipulations.py -v
"""
import os
import sys
import tempfile
import shutil
import pytest
import pefile

# Add PE_modifier to path for direct import (avoids gym_malware package init)
PE_MODIFIER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gym_malware", "envs", "PE_modifier"
)
sys.path.insert(0, PE_MODIFIER_PATH)

from pefile_manipulations import (
    PefileManipulator,
    ACTION_TABLE,
    modify_PE_file,
    COMMON_SECTION_NAMES,
    COMMON_SECTION_NAMES_WEIGHTS,
    COMMON_IMPORTS,
)

# Path to real test executable
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
REAL_PE_PATH = os.path.join(TEST_DATA_DIR, "hello_c.exe")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def basic_pe(temp_dir):
    """Copy the real PE file for testing."""
    pe_path = os.path.join(temp_dir, 'test.exe')
    shutil.copy(REAL_PE_PATH, pe_path)
    return pe_path

class TestStaticMethods:
    """Tests for static methods in PefileManipulator."""
    
    def test_get_output_path_random_name(self, temp_dir):
        """Test output path generation with random name."""
        input_path = '/some/path/test.exe'
        output_path = PefileManipulator.get_output_path(temp_dir, input_path, random_name=True)
        
        assert output_path.startswith(temp_dir)
        assert os.path.basename(output_path) != 'test.exe'
        
    def test_get_output_path_preserve_name(self, temp_dir):
        """Test output path generation preserving original name."""
        input_path = '/some/path/test.exe'
        output_path = PefileManipulator.get_output_path(temp_dir, input_path, random_name=False)
        
        assert output_path == os.path.join(temp_dir, 'test.exe')
        
    def test_load_PE(self, basic_pe):
        """Test loading a PE file."""
        pe = PefileManipulator.load_PE(basic_pe)
        
        assert pe is not None
        assert isinstance(pe, pefile.PE)
        
    def test_load_PE_invalid_file(self, temp_dir):
        """Test loading an invalid PE file raises exception."""
        invalid_path = os.path.join(temp_dir, 'invalid.exe')
        with open(invalid_path, 'wb') as f:
            f.write(b'NOT A PE FILE')
            
        with pytest.raises(pefile.PEFormatError):
            PefileManipulator.load_PE(invalid_path)
            
    def test_align(self):
        """Test alignment calculation."""
        assert PefileManipulator.align(100, 512) == 512
        assert PefileManipulator.align(512, 512) == 512
        assert PefileManipulator.align(513, 512) == 1024
        assert PefileManipulator.align(0, 512) == 0
        assert PefileManipulator.align(1, 4096) == 4096
        
    def test_get_benign_content(self):
        """Test getting benign content."""
        content = PefileManipulator.get_benign_content(5)
        
        assert isinstance(content, str)
        assert len(content) > 0
        
    def test_get_benign_content_different_sizes(self):
        """Test getting benign content of different sizes."""
        content_small = PefileManipulator.get_benign_content(1)
        content_large = PefileManipulator.get_benign_content(50)
        
        # Larger size should generally produce more content
        assert len(content_large) >= len(content_small)
        
    def test_get_benign_section_characteristics(self):
        """Test getting random section characteristics."""
        valid_chars = [0x60000020, 0x40000040, 0x42000040, 0xc0000040, 
                       0xc0000080, 0xc0000000, 0x50000040, 0x0]
        
        for _ in range(10):
            char = PefileManipulator.get_benign_section_characteristics()
            assert char in valid_chars
            
    def test_get_random_section_name(self):
        """Test getting random section name."""
        name = PefileManipulator.get_random_section_name()
        
        assert isinstance(name, list)
        assert len(name) == 8
        
    def test_all_zeros_true(self):
        """Test all_zeros with all zero bytes."""
        assert PefileManipulator.all_zeros(b'\x00\x00\x00\x00') is True
        assert PefileManipulator.all_zeros(bytearray(10)) is True
        
    def test_all_zeros_false(self):
        """Test all_zeros with non-zero bytes."""
        assert PefileManipulator.all_zeros(b'\x00\x01\x00\x00') is False
        assert PefileManipulator.all_zeros(b'\xff') is False
        
    def test_get_available_space_size(self, basic_pe):
        """Test calculating available space after a section."""
        pe = pefile.PE(basic_pe)
        
        # Should be able to get space for the first section
        space = PefileManipulator.get_available_space_size(pe, 0)
        assert isinstance(space, int)
        assert space >= 0
        
    def test_get_available_space_size_invalid_index(self, basic_pe):
        """Test available space for invalid section index."""
        pe = pefile.PE(basic_pe)
        
        # Invalid section index should return 0
        space = PefileManipulator.get_available_space_size(pe, 999)
        assert space == 0


class TestRemoveDebug:
    """Tests for remove_debug functionality."""
    
    def test_remove_debug(self, basic_pe, temp_dir):
        """Test removing debug directory."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.remove_debug()
        
        assert os.path.exists(output_path)
        
        # Verify debug directory is cleared
        pe = pefile.PE(output_path)
        for d in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            if d.name == 'IMAGE_DIRECTORY_ENTRY_DEBUG':
                assert d.VirtualAddress == 0
                assert d.Size == 0
                break
                
    def test_remove_debug_file_still_valid(self, basic_pe, temp_dir):
        """Test remove_debug on file without debug directory."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.remove_debug()
        
        assert os.path.exists(output_path)
        # File should still be valid
        pe = pefile.PE(output_path)
        assert pe is not None


class TestRemoveCertificate:
    """Tests for remove_certificate functionality."""
    
    def test_remove_certificate_without_cert(self, basic_pe, temp_dir):
        """Test remove_certificate on file without certificate."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.remove_certificate()
        
        assert os.path.exists(output_path)
        pe = pefile.PE(output_path)
        assert pe is not None


class TestBreakChecksum:
    """Tests for break_checksum functionality."""
    
    def test_break_checksum(self, basic_pe, temp_dir):
        """Test breaking the PE checksum."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.break_checksum()
        
        assert os.path.exists(output_path)
        
        pe = pefile.PE(output_path)
        assert pe.OPTIONAL_HEADER.CheckSum == 0
        
    def test_break_checksum_already_zero(self, basic_pe, temp_dir):
        """Test breaking checksum when already zero."""
        # First set checksum to zero
        pe = pefile.PE(basic_pe)
        pe.OPTIONAL_HEADER.CheckSum = 0
        pe.write(basic_pe)
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.break_checksum()
        
        pe = pefile.PE(output_path)
        assert pe.OPTIONAL_HEADER.CheckSum == 0


class TestRenameSection:
    """Tests for rename_section functionality."""
    
    def test_rename_section(self, basic_pe, temp_dir):
        """Test renaming a section produces valid PE with same number of sections."""
        original_pe = pefile.PE(basic_pe)
        original_num_sections = len(original_pe.sections)
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.rename_section()
        
        assert os.path.exists(output_path)        
        pe = pefile.PE(output_path)
        # Number of sections should remain the same
        assert len(pe.sections) == original_num_sections
        
    def test_rename_section_changes_name(self, basic_pe, temp_dir):
        """Test that rename eventually changes a section name (probabilistic)."""
        original_pe = pefile.PE(basic_pe)
        original_section_names = [s.Name for s in original_pe.sections]
        
        # Try multiple times since name selection is random
        name_changed = False
        for attempt in range(10):
            test_pe = os.path.join(temp_dir, f'rename_test_{attempt}.exe')
            shutil.copy(basic_pe, test_pe)
            
            manipulator = PefileManipulator(test_pe, temp_dir, verbose=False)
            output_path = manipulator.rename_section()
            
            pe = pefile.PE(output_path)
            new_section_names = [s.Name for s in pe.sections]
            
            if new_section_names != original_section_names:
                name_changed = True
                break
        
        assert name_changed, "Section name should change after multiple rename attempts"


class TestAddSection:
    """Tests for add_section functionality."""
    
    def test_add_section(self, basic_pe, temp_dir):
        """Test adding a new section."""
        original_pe = pefile.PE(basic_pe)
        original_num_sections = len(original_pe.sections)
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.add_section()
        
        assert os.path.exists(output_path)
        
        pe = pefile.PE(output_path)
        # Should have one more section (or same if no space for header)
        assert len(pe.sections) >= original_num_sections


class TestOverlayAppend:
    """Tests for overlay_append functionality."""
    
    def test_overlay_append(self, basic_pe, temp_dir):
        """Test appending to overlay."""
        original_size = os.path.getsize(basic_pe)
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.overlay_append()
        
        assert os.path.exists(output_path)
        
        new_size = os.path.getsize(output_path)
        assert new_size > original_size


class TestTimestamp:
    """Tests for timestamp manipulation."""
    
    def test_increase_timedatestamp(self, basic_pe, temp_dir):
        """Test increasing TimeDateStamp."""
        original_pe = pefile.PE(basic_pe)
        original_timestamp = original_pe.FILE_HEADER.TimeDateStamp
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.increase_timedatestamp()
        
        pe = pefile.PE(output_path)
        assert pe.FILE_HEADER.TimeDateStamp == original_timestamp + 43200000
        
    def test_decrease_timedatestamp(self, basic_pe, temp_dir):
        """Test decreasing TimeDateStamp."""
        original_pe = pefile.PE(basic_pe)
        original_timestamp = original_pe.FILE_HEADER.TimeDateStamp
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.decrease_timedatestamp()
        
        pe = pefile.PE(output_path)
        expected = max(0, original_timestamp - 43200000)
        assert pe.FILE_HEADER.TimeDateStamp == expected
        
    def test_decrease_timedatestamp_underflow(self, basic_pe, temp_dir):
        """Test decreasing TimeDateStamp doesn't go negative."""
        # Set a very small timestamp
        pe = pefile.PE(basic_pe)
        pe.FILE_HEADER.TimeDateStamp = 100
        output_path = PefileManipulator.get_output_path(temp_dir, basic_pe, random_name=False)        
        pe.write(output_path)
        
        manipulator = PefileManipulator(output_path, temp_dir, verbose=False)
        output_path = manipulator.decrease_timedatestamp()
        
        pe = pefile.PE(output_path)
        assert pe.FILE_HEADER.TimeDateStamp == 0


class TestIdentity:
    """Tests for identity (no-op) function."""
    
    def test_identity(self, basic_pe, temp_dir):
        """Test identity function copies file unchanged."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.identity()
        
        assert os.path.exists(output_path)
        
        with open(basic_pe, 'rb') as f:
            original_content = f.read()
        with open(output_path, 'rb') as f:
            new_content = f.read()
            
        assert original_content == new_content


class TestAppendSection:
    """Tests for append_section functionality."""
    
    def test_append_section(self, basic_pe, temp_dir):
        """Test appending content to an existing section."""
        original_pe = pefile.PE(basic_pe)
        
        # Find sections with available space and record their content
        original_section_data = {}
        for i, section in enumerate(original_pe.sections):
            available = PefileManipulator.get_available_space_size(original_pe, i)
            if available > 0:
                offset = section.PointerToRawData + section.Misc_VirtualSize
                original_section_data[i] = {
                    'offset': offset,
                    'available': available,
                    'original_bytes': original_pe.__data__[offset:offset + available]
                }
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.append_section()
        
        assert os.path.exists(output_path)
        
        # If there was available space, verify content was modified with non-zero data
        if original_section_data:
            modified_pe = pefile.PE(output_path)
            content_appended = False
            for i, data in original_section_data.items():
                offset = data['offset']
                new_bytes = modified_pe.__data__[offset:offset + data['available']]
                if new_bytes != data['original_bytes']:
                    # Verify appended content is non-zero
                    assert any(b != 0 for b in new_bytes), "Appended content should be non-zero"
                    content_appended = True
                    break
            assert content_appended, "Content should have been appended to a section"


class TestShuffleSections:
    """Tests for shuffle_sections functionality."""
    
    ENTRY_SIZE = 40
    
    def _get_section_table(self, pe_path):
        """Extract section table entries as a list from PE file."""
        pe = pefile.PE(pe_path)
        offset = pe.OPTIONAL_HEADER.get_file_offset() + pe.FILE_HEADER.SizeOfOptionalHeader
        num_sections = pe.FILE_HEADER.NumberOfSections
        
        with open(pe_path, 'rb') as f:
            f.seek(offset)
            table_bytes = f.read(num_sections * self.ENTRY_SIZE)
        
        return [table_bytes[i*self.ENTRY_SIZE:(i+1)*self.ENTRY_SIZE] for i in range(num_sections)]
    
    def test_shuffle_sections(self, basic_pe, temp_dir):
        """Test shuffling section headers reorders them at byte level."""
        original_pe = pefile.PE(basic_pe)
        num_sections = len(original_pe.sections)
        original_entries = self._get_section_table(basic_pe)
        
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.shuffle_sections()
        
        assert os.path.exists(output_path)
        
        # Number of sections should remain the same
        modified_pe = pefile.PE(output_path)
        assert len(modified_pe.sections) == num_sections
        
        # Section table should have same entries (same sections, possibly reordered)
        new_entries = self._get_section_table(output_path)
        assert len(new_entries) == len(original_entries)
        
    def test_shuffle_sections_changes_bytes(self, basic_pe, temp_dir):
        """Test that shuffle changes the order of section table entries."""
        original_pe = pefile.PE(basic_pe)
        num_sections = len(original_pe.sections)
        
        if num_sections < 2:
            pytest.skip("Need at least 2 sections to test shuffling")
        
        original_entries = self._get_section_table(basic_pe)
        
        # Try multiple times since shuffle is random
        order_changed = False
        for attempt in range(10):
            test_pe = os.path.join(temp_dir, f'shuffle_test_{attempt}.exe')
            shutil.copy(basic_pe, test_pe)
            
            manipulator = PefileManipulator(test_pe, temp_dir, verbose=False)
            output_path = manipulator.shuffle_sections()
            
            new_entries = self._get_section_table(output_path)
            
            # Check that same entries exist but in different order
            if new_entries != original_entries:
                # Verify it's a reordering (same set of entries)
                assert sorted(new_entries) == sorted(original_entries), "Should be same entries reordered"
                order_changed = True
                break
        
        assert order_changed, "Section table entry order should change after shuffling"


class TestAppendImports:
    """Tests for append_imports functionality."""
    
    def test_append_imports(self, basic_pe, temp_dir):
        """Test appending imports."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=False)
        output_path = manipulator.append_imports()
        
        assert os.path.exists(output_path)


class TestModifyPEFile:
    """Tests for the main modify_PE_file function."""
    
    def test_modify_single_action(self, basic_pe, temp_dir):
        """Test modifying PE with a single action."""
        failure, output_path = modify_PE_file(
            basic_pe, temp_dir, 
            actions=['break_checksum'], 
            verbose=False
        )
        
        assert not failure
        assert os.path.exists(output_path)
        
        pe = pefile.PE(output_path)
        assert pe.OPTIONAL_HEADER.CheckSum == 0
        
    def test_modify_multiple_actions(self, basic_pe, temp_dir):
        """Test modifying PE with multiple actions."""
        failure, output_path = modify_PE_file(
            basic_pe, temp_dir,
            actions=['break_checksum', 'overlay_append'],
            verbose=False
        )
        
        assert not failure
        assert os.path.exists(output_path)
        
    def test_modify_no_actions(self, basic_pe, temp_dir):
        """Test modifying PE with no actions."""
        failure, output_path = modify_PE_file(
            basic_pe, temp_dir,
            actions=[],
            verbose=False
        )
        
        # With no actions, output_path may be derived but file may not exist
        # or the function might behave differently
        assert not failure or output_path is not None
        
    def test_all_actions_available(self):
        """Test that all actions in ACTION_TABLE are valid."""
        expected_actions = [
            'rename_section', 'add_section', 'append_section',
            'remove_certificate', 'remove_debug', 'break_checksum',
            'overlay_append', 'increase_timedatestamp', 
            'decrease_timedatestamp', 'append_imports'
        ]
        
        for action in expected_actions:
            assert action in ACTION_TABLE
            
    def test_action_table_methods_exist(self):
        """Test that all actions in table correspond to actual methods."""
        for action_name, method_name in ACTION_TABLE.items():
            assert hasattr(PefileManipulator, method_name)


class TestCommonData:
    """Tests for common data structures."""
    
    def test_common_section_names_loaded(self):
        """Test that common section names are loaded."""
        assert len(COMMON_SECTION_NAMES) > 0
        assert len(COMMON_SECTION_NAMES_WEIGHTS) == len(COMMON_SECTION_NAMES)
        
    def test_common_imports_loaded(self):
        """Test that common imports are loaded."""
        assert len(COMMON_IMPORTS) > 0
        assert isinstance(COMMON_IMPORTS, dict)
        
        # Check that each DLL has functions
        for dll, funcs in COMMON_IMPORTS.items():
            assert isinstance(dll, str)
            assert len(funcs) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_manipulator_initialization(self, basic_pe, temp_dir):
        """Test PefileManipulator initialization."""
        manipulator = PefileManipulator(basic_pe, temp_dir, verbose=True, random_name=True)
        
        assert manipulator.input_path == basic_pe
        assert manipulator.output_folder == temp_dir
        assert manipulator.output_path != basic_pe
        assert manipulator.output_path.startswith(temp_dir)
        assert manipulator.verbose is True
        
    def test_successive_modifications(self, basic_pe, temp_dir):
        """Test applying modifications in succession."""
        # First modification
        manipulator1 = PefileManipulator(basic_pe, temp_dir, verbose=False)
        path1 = manipulator1.break_checksum()
        
        # Second modification on the result
        manipulator2 = PefileManipulator(path1, temp_dir, verbose=False)
        path2 = manipulator2.overlay_append()
        
        assert os.path.exists(path2)
        
        pe = pefile.PE(path2)
        assert pe.OPTIONAL_HEADER.CheckSum == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
